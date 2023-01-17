import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

from torch import nn
import torch.utils.data
tsla_df = pd.read_csv('/kaggle/input/tesla-stock-data-from-2010-to-2020/TSLA.csv', index_col='Date', parse_dates=['Date'])
tsla_df = tsla_df[['Open', 'High', 'Low', 'Volume', 'Close']]
print(tsla_df.info())
print(tsla_df.describe())
def load_data(df, seq_len, out_feature=5, train_ratio=0.8, is_test=False):
    scaler = StandardScaler()
    scaler.fit(df)

    train_norm = scaler.transform(df)
    data = []
    for index in range(len(train_norm) - seq_len):
        # create all possible sequences
        data.append(train_norm[index:index + seq_len])

    data = np.array(data)

    train_len = len(data) if is_test else int(train_ratio * len(data))

    # train_x are sequences of seq_len-1 days. Features of each day are OPEN, CLOSE, HIGH, LOW, VOLUME
    # train_y is CLOSE price of day seq_len
    # shape is (n_data, n_sequence, features)
    train_x = data[:train_len, :-1, :]
    train_y = data[:train_len, -1, -out_feature:]

    val_x = data[train_len:, :-1, :]
    val_y = data[train_len:, -1, -out_feature:]

    return train_x, train_y, val_x, val_y
seq_len = 100
batch_size = 128
n_epoch = 50
n_feature = 5
out_feature = 1

train_df = tsla_df.loc[:'2018']
test_df = tsla_df.loc['2019':]

train_x, train_y, val_x, val_y = load_data(train_df, seq_len, out_feature=out_feature)
train_x = torch.from_numpy(train_x).float().cuda()
train_y = torch.from_numpy(train_y).float().cuda()
val_x = torch.from_numpy(val_x).float().cuda()
val_y = torch.from_numpy(val_y).float().cuda()

train = torch.utils.data.TensorDataset(train_x, train_y)
val = torch.utils.data.TensorDataset(val_x, val_y)

train_loader = torch.utils.data.DataLoader(dataset=train,
                                               batch_size=batch_size,
                                               shuffle=True, drop_last=True)

val_loader = torch.utils.data.DataLoader(dataset=val,
                                             batch_size=1,
                                             shuffle=False)
class StockRNN(nn.Module):

    def __init__(self, n_feature=5, out_feature=5, n_hidden=256, n_layers=2, drop_prob=0.5):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_feature = n_feature

        self.lstm = nn.LSTM(self.n_feature, self.n_hidden, self.n_layers, dropout=self.drop_prob, batch_first=True)

        self.dropout = nn.Dropout(drop_prob)

        self.fc = nn.Linear(n_hidden, out_feature)

    def forward(self, x, hidden):
        # x.shape (batch, seq_len, n_features)
        l_out, l_hidden = self.lstm(x, hidden)

        # out.shape (batch, seq_len, n_hidden*direction)
        out = self.dropout(l_out)

        # out.shape (batch, out_feature)
        out = self.fc(out[:, -1, :])

        # return the final output and the hidden state
        return out, l_hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        return hidden
net = StockRNN(n_feature=n_feature, out_feature=out_feature)
net.cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters())
val_loss_list = []
for epoch in range(n_epoch):

    for i, (x, y) in enumerate(train_loader):
        output, hidden = net(x, net.init_hidden(batch_size))
        loss = criterion(output, y)

        net.zero_grad()
        loss.backward()
        optimizer.step()

    net.eval()
    val_loss_sum = 0
    for i, (x, y) in enumerate(val_loader):
        with torch.no_grad():
            output, hidden = net(x, net.init_hidden(1))
            val_loss = criterion(output, y)
            val_loss_sum += val_loss.item()

    val_loss_list.append(val_loss_sum/len(val_loader))
    print('End of Epoch ', epoch, 'Val loss: ', val_loss_sum/len(val_loader))
    net.train()

plt.plot(val_loss_list)
# test_df.shape
test_x, test_y, _, _ = load_data(test_df, seq_len, out_feature=out_feature, is_test=True)

test_x = torch.from_numpy(test_x).float().cuda()
test_y = torch.from_numpy(test_y).float().cuda()

test = torch.utils.data.TensorDataset(test_x, test_y)

test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=1, shuffle=False)

net.eval()
test_predict = []
for i, (x, y) in enumerate(test_loader):
    with torch.no_grad():
        output, hidden = net(x, net.init_hidden(1))
    test_predict.append(output[:, -1])

plt.plot(test_y[:, -1].cpu().numpy(), label='GT')
plt.plot(test_predict, label='Predict')
plt.legend()