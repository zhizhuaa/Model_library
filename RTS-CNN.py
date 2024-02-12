import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from Embed import DataEmbedding
from convnext import Block
from CBAM import CBAMBlock


def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.conv = nn.Sequential(
            Block(1),
            Block(1),
            Block(1)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)

        return res


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.enc_embedding1 = DataEmbedding(1, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.predict_linear = nn.Linear(
            self.seq_len, self.pred_len + self.seq_len)
        self.conv = nn.Sequential(
            Block(1),
            Block(1),
            Block(1)
        )
        self.cbam = CBAMBlock(channel=2, reduction=2, kernel_size=configs.seq_len + configs.pred_len)
        self.mlp = nn.Linear(2,1)

    def forecast(self, x_enc, x_mark_enc):
        #  Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        #  The first strategy
        x_enc1 = x_enc[:, :, -1].unsqueeze(-1)
        enc_out = self.enc_embedding1(x_enc1, x_mark_enc)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)
        for i in range(self.layer):
            enc_out = self.model[i](enc_out)
        dec_out = self.projection(enc_out)
        #  The second strategy
        enc_out1 = self.enc_embedding(x_enc, x_mark_enc)
        enc_out1 = self.predict_linear(enc_out1.permute(0, 2, 1)).permute(0, 2, 1)
        enc_out1 = enc_out1.unsqueeze(1)
        enc_out1 = self.conv(enc_out1).squeeze(1)
        dec_out1 = self.projection(enc_out1)

        dec_out2 = torch.cat((dec_out, dec_out1), -1)
        dec_out2 = dec_out2.permute(0, 2, 1).unsqueeze(-1)
        dec_out2 = self.cbam(dec_out2)
        dec_out2 = dec_out2.squeeze(-1).permute(0, 2, 1)
        dec_out2 = self.mlp(dec_out2)
        #  De-Normalization
        dec_out2 = dec_out2 * \
                  (stdev[:, 0, -1].unsqueeze(1).unsqueeze(-1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out2 = dec_out2 + \
                  (means[:, 0, -1].unsqueeze(1).unsqueeze(-1).repeat(
                      1, self.pred_len + self.seq_len, 1))

        return dec_out2


