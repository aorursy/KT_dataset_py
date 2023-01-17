import warnings

warnings.filterwarnings('ignore')



import torch

from torch import nn



import numpy as np



from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go
t = np.arange(0,30,0.01)

len(t)*0.8
end = [np.sin(1000), np.cos(1000)]
x = []

for i in t:

    x.append(np.array([np.sin(i), np.sin(i+0.5)]))

X = np.array(x)

y = []

for i in range(len(X)):

    try:

        y_1 = 0

        y_2 = 0

        y_1 += X[i-1][0] - X[i-1][1] + X[i-5][0] - 2*X[i-8][1] + X[i-12][1]

        y_2 += X[i-1][0] + X[i-1][1] - 2*X[i-4][0] - 4*X[i-8][1] + 3*X[i-10][1]

    except:

        y_1 = end[0]

        y_2 = end[1]

    tmp_y = np.array([y_1, y_2])

    y.append(tmp_y)

y = np.array(y)

print(len(X),len( y))
X_train, X_test = X[20:1500], X[1500:3000]

y_train, y_test = y[20:1500], y[1500:3000]
plt.plot(X_train)
plt.plot(y_train)
def get_data(i):

    tmp_x = X_train[i : i+20]

    tmp_y = y_train[i : i+20]

    tmp_x = torch.from_numpy(tmp_x).float().unsqueeze(0)

    tmp_y = torch.from_numpy(tmp_y).float().unsqueeze(0)

    return tmp_x, tmp_y

def get_data_val(i):

    tmp_x = X_test[i : i+20]

    tmp_y = y_test[i : i+20]

    tmp_x = torch.from_numpy(tmp_x).float().unsqueeze(0)

    tmp_y = torch.from_numpy(tmp_y).float().unsqueeze(0)

    return tmp_x, tmp_y
"""

sclr_trn = MinMaxScaler()

sclr_val = MinMaxScaler()

X_train = sclr_trn.fit_transform(X_train)

y_train = sclr_val.fit_transform(y_train)

X_test = sclr_trn.transform(X_test)

y_test = sclr_val.transform(y_test)



plt.plot(X_train)

plt.show()

plt.plot(X_test)

plt.show()



plt.plot(y_train)

plt.show()



plt.plot(y_test)

plt.show()

"""
class Sequence(nn.Module):

    def __init__(self):

        super(Sequence, self).__init__()

        self.lstm1 = nn.LSTM(2, 2, 3)

        self.linear = nn.Linear(2, 2)

        

    def forward(self, x):

        out, _ = self.lstm1(x)

        #

        #out = self.linear(out)

        #out = nn.ReLU()(out)

        #out = nn.Linear(2,2)(out)

        #out = nn.Sigmoid()(out)

        return out
net = Sequence()

#nn.LSTM(2, 2, 3, batch_first=True)

optim = torch.optim.Adam(net.parameters(), lr=1e-4)

criterion = nn.MSELoss()
for _ in range(100):

    avg = 0

    for i in range(len(X_train)):

        optim.zero_grad()

        t_x, target = get_data(i)

        out=net(t_x)

        loss = criterion(out, target)

        loss.backward()

        optim.step()

        avg += loss.item()/len(X_train)

    avg_val = 0

    for i in range(len(X_test)):

        t_x, target = get_data_val(i)

        out =net(t_x)

        loss = criterion(out, target)

        avg_val += loss.item()/len(X_test)

    print(avg, ' : ', avg_val)
pred = []

net.eval()

for i in range(len(X_test)):

    t_x, target = get_data_val(i)

    out = net(t_x)

    pred.append(np.array(out.tolist()[0][-1]))
pred = torch.tensor(pred).float()

len(pred), len(y_test)
y_test_1 = torch.from_numpy(y_test).float()

pred = np.array(pred)
plt.plot(pred[:,0], c='r')

plt.plot(y_test[:,0], c='b')
plt.plot(pred[:,1], c='r')

plt.plot(y_test[:,1], c='b')