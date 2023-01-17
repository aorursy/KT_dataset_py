# Install fastai2 (note that soon fastai2 will be officially released as fastai)

!pip install fastai2
import numpy as np

import scipy.stats

import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_pacf

import torch

import torch.nn as nn

import torch.optim as optim

from tqdm import tqdm

from IPython.core.debugger import set_trace

from fastai2.vision.all import *
N  = 20000



b   = 0.1

c   = 0.2

tau = 17



y = [0.9697, 0.9699, 0.9794, 1.0003, 1.0319, 1.0703, 1.1076, 1.1352, 1.1485,

     1.1482, 1.1383, 1.1234, 1.1072, 1.0928, 1.0820, 1.0756, 1.0739, 1.0759]



for n in range(17,N+99):

    y.append(y[n] - b*y[n] + c*y[n-tau]/(1+y[n-tau]**10))

y = y[100:]
plt.figure(figsize=(5,4), dpi=150)

plt.plot(y[:500])
sl = int(N//1.3333333333)

y_train, y_valid = np.array(y[:sl]), np.array(y[sl:])

y_train.shape, y_valid.shape
trn_len, pred_len = 100, 400
def create_sequences(yin, input_seq_size, output_seq_size):

    xout = []

    yout = []

    for ii in tqdm(range(yin.shape[0]-input_seq_size-output_seq_size)):

        xout.append(yin[ii:ii+input_seq_size, ...].view(1, 1, -1))

        yout.append(yin[ii+input_seq_size:ii+input_seq_size+output_seq_size, ...].view(1, 1, -1))

    xout = torch.cat(xout, dim=0)

    yout = torch.cat(yout, dim=0)

    return xout, yout.squeeze()
x_train, y_train = create_sequences(torch.from_numpy(y_train).float(), trn_len, pred_len)

x_valid, y_valid = create_sequences(torch.from_numpy(y_valid).float(), trn_len, pred_len)

x_train.size(), y_train.size(), x_valid.size(), y_valid.size()
train_ind = list(range(len(x_train)))

valid_ind = list(range(len(x_train), len(x_train)+len(x_valid)))

x = torch.cat([x_train, x_valid], dim=0)

y = torch.cat([y_train, y_valid], dim=0)

x.size(), y.size()
data = Datasets(list(range(len(x))), [lambda i : x[i], lambda i : y[i]], splits=[train_ind, valid_ind])

dls = data.dataloaders(bs=64)
class TimeSeriesModel(nn.Module):

    def __init__(self, input_size, output_size):

        super(TimeSeriesModel, self).__init__()

        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=7, stride=2)

        self.conv1_bn = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2)

        self.conv2_bn = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=2)

        self.conv3_bn = nn.BatchNorm1d(256)

        self.drop = nn.Dropout(0.5)

        self.pool = nn.AdaptiveAvgPool1d(10)

        self.linear = nn.Linear(10*256, output_size)

        self.linear_bn = nn.BatchNorm1d(output_size)

        self.out = nn.Linear(output_size, output_size)



    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = self.conv1_bn(x)

        x = F.relu(self.conv2(x))

        x = self.conv2_bn(x)

        x = F.relu(self.conv3(x))

        x = self.conv3_bn(x)

        x = self.pool(x)

        x = x.view(-1, 10*256)

        x = F.relu(self.linear(x))

        x = self.drop(self.linear_bn(x))

        return self.out(x)
model = TimeSeriesModel(1, pred_len)

learn = Learner(dls, model, loss_func=nn.MSELoss())
learn.lr_find()
learn.fit_one_cycle(20, max_lr=3e-2)
ye_valid, y_valid = learn.get_preds()

ye_valid.shape, y.shape
fig, axes = plt.subplots(ncols=4, nrows=3, figsize=(12,6), dpi=150)

for i, ax in enumerate(axes.flat):

    plot_idx = np.random.choice(np.arange(0, len(ye_valid)))

    true = np.concatenate([x_valid.numpy()[plot_idx,-1,:].reshape(-1), y_valid.numpy()[plot_idx,:].reshape(-1)])

    pred = np.concatenate([x_valid.numpy()[plot_idx,-1,:].reshape(-1), ye_valid[plot_idx,:].reshape(-1)])

    ax.plot(pred, color='red', label='preds')

    ax.plot(true, color='green', label='true')

    ax.vlines(trn_len-1, np.min(true), np.max(true), color='black')

    if i == 0: ax.legend()

fig.tight_layout();
plt.figure(figsize=(5,4), dpi=150)

plt.plot(((ye_valid-y_valid)**2).mean(0))

plt.ylabel('MSE')