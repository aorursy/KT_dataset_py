# neural network in pytorch

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from math import sqrt, exp, pi, floor



torch.manual_seed(1)
class Data(Dataset):

    # Constructor

    def __init__(self, x, y):

        self.x = torch.Tensor(np.asarray(x).astype(np.float16))

        # define output

        self.y = torch.Tensor(y.reshape([-1, 1]).astype(np.float16))

        self.len = self.x.shape[0]



    # Getter

    def __getitem__(self, index):

        return self.x[index], self.y[index]



    # Get Length

    def __len__(self):

        return self.len





class Net(nn.Module):

    def __init__(self, dropout_p, signallength):

        super(Net, self).__init__()

        self.drop = nn.Dropout(p=dropout_p)

        self.batchnorm1 = nn.BatchNorm1d(num_features=signallength)

        self.CNN1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=30, stride=5)

        self.maxpool1 = nn.MaxPool1d(kernel_size=5, stride=2)

        self.CNN2 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=30, stride=5)

        self.maxpool2 = nn.MaxPool1d(kernel_size=5, stride=2)

        self.sigmoid = nn.Sigmoid()

        self.relu = nn.functional.leaky_relu

        self.linear = nn.Linear(16*11, 100)

        self.linear2 = nn.Linear(100, 10)

        self.linear3 = nn.Linear(10, 1)





    def forward(self, activation):

        activation = activation.view([activation.shape[0], 1, activation.shape[1]])

        activation = self.CNN1(activation)

        # print(activation.shape)

        activation = self.maxpool1(activation)

        # print(activation.shape)

        # activation = self.relu(activation)

        # print(activation.shape)

        activation = self.CNN2(activation)

        # print(activation.shape)

        activation = self.maxpool2(activation)

        # print(activation.shape)

        # activation = self.relu(activation)

        # print(activation.shape)

        activation = activation.view(-1, 16*11)

        # print(activation.shape)

        activation = self.relu(self.linear(activation), negative_slope=0.1)

        # print(activation.shape)

        activation = self.relu(self.linear2(activation), negative_slope=0.1)

        activation = self.sigmoid(self.linear3(activation))

        return activation
def accuracy(model, data_set, test_set, epoch):

    model.eval()

    mr = model(data_set.x).detach().numpy()

    mt = model(test_set.x).detach().numpy()

    yr = data_set.y.detach().numpy()

    yt = test_set.y.detach().numpy()



    error_r = np.abs((yr - np.round(mr, 0))).mean()

    error_t = np.abs((yt - np.round(mt, 0))).mean()

    mean_positive = mr[yr == 1].mean()

    mean_negative = mr[yr == 0].mean()

    ratio = mean_positive/mean_negative

    model.train()

    if (epoch)%300 == 0:

        plt.figure(1)

        plt.subplot(211)

        plt.plot(mt, '-o')

        plt.subplot(212)

        plt.plot(mr, '-o')

        plt.title(str(epoch))

        plt.savefig(str(epoch) + '.png')

        plt.close()

    return error_r, mean_positive, mean_negative, ratio, error_t
def train(f, data_set, test_set, model, criterion, train_loader, optimizer, epochs=100):

    LOSS = []

    ACC_train = []

    ACC_test = []

    for epoch in range(epochs):

        for x, y in train_loader:

            optimizer.zero_grad()

            yhat = model(x)

            loss = criterion(yhat, y)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        LOSS.append(loss.item())

        error_r, mean_positive, mean_negative, ratio, error_t = accuracy(model, data_set, test_set, epoch)

        ACC_train.append(1 - error_r)

        ACC_test.append(1 - error_t)

        newstr = ','.join([str(epoch), str(error_r),str(mean_positive),str(mean_negative), str(ratio),str(error_t), str(loss.item())])

        f.write(newstr + '\n')

        if epoch % 500 == 0:

            print(newstr)

            for param_group in optimizer.param_groups:

                param_group['lr'] = param_group['lr']*.99



    # Final Plot for results

    from sklearn.metrics import precision_recall_curve



    y_true = test_set.y.detach().numpy()

    model.eval()

    y_scores = model(test_set.x).detach().numpy()

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    fig, ax1 = plt.subplots()

    color = 'tab:red'

    ax1.plot(LOSS, color=color)

    ax1.set_xlabel('Iteration', color=color)

    ax1.set_ylabel('total loss', color=color)

    ax1.set_ylim([0,0.1])

    ax1.tick_params(axis='y', color=color)



    ax2 = ax1.twinx()

    color = 'tab:blue'

    ax2.set_ylabel('accuracy', color=color)  # we already handled the x-label with ax1

    ax2.plot(ACC_test, color=color)

    ax2.tick_params(axis='y', color=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.title('Precision: {:0.2f}, Recall: {:0.2f}'.format(precision[0], recall[0]))

    plt.savefig('Percision-Recall.png')

    plt.close()

    return LOSS

def gauss(n=11,sigma=1):

    r = range(-int(n/2),int(n/2)+1)

    return [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]



def format_into_xy(df):

    gfilt = np.asarray(gauss())

    rowi = df.loc[0, :]._ndarray_values[1:]

    rowi = np.convolve(rowi, gfilt)

    xfft = np.abs(np.fft.fft(rowi)[:floor(len(rowi) / 2)])

    # xfft= rowi

    arr_len = xfft.shape[0]

    arr_wid = len(df)

    sl = [] # Use this to delete values from the data set if there is a problem with them

    y = np.zeros([arr_wid - len(sl), 1])

    x = np.zeros([arr_wid - len(sl), arr_len])

    skip = 0

    for i in range(arr_wid):

        if i in sl:

            print('skip')

            skip +=1

        else:

            rowi = df.loc[i, :]._ndarray_values[1:]

            rowi= np.convolve(rowi, gfilt)

            xfft = np.abs(np.fft.fft(rowi)[:floor(len(rowi) / 2)])

            x[i - skip, :] = ((xfft - np.mean(xfft)) / np.std(xfft))

            y[i - skip] = df.loc[i - skip, :]._ndarray_values[0]

            if y[i - skip] == 2:



                y[i - skip] = 1

            else:

                y[i - skip] = 0

    return x, y

import os

print(os.listdir("../input"))

df = pd.read_csv('../input/exoTrain.csv', header=0, index_col=False)

df2 = pd.read_csv('../input/exoTest.csv', header=0, index_col=False)

x, y = format_into_xy(df)

x2, y2 = format_into_xy(df2)

f = open('log.csv', 'w+')

for col in ['margin']:

    data_set = Data(x, y)

    test_set = Data(x2, y2)

    train_loader = DataLoader(dataset=data_set, batch_size=data_set.len)

    criterion = nn.BCELoss()

    learning_rate = 0.1

    model = Net(0.05, x.shape[1])

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)

    model.train()

    LOSS = train(f, data_set, test_set, model, criterion, train_loader, optimizer, epochs=4000)

f.close()