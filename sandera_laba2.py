# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd



data = pd.read_csv('../input/train.csv')

data.head()
import torch



x = torch.ones(2, 2, requires_grad=True)

y = x + 2

z = y * y * 3

out = z.mean()

out.backward()

x.grad
data.shape
import torch.nn as nn

import torch.nn.functional as F



class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.fc1 = nn.Linear(28 * 28, 800)

        self.fc2 = nn.Linear(800, 200)

        self.fc3 = nn.Linear(200, 10)

    

    def forward(self, x):

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return F.log_softmax(x, dim=1)





net = Net()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net.cuda(device=device)

net = Net()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net.cuda(device=device)
import pycuda.driver as cuda

cuda.init()
import torch.optim as optim



optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

# create a loss function

criterion = nn.NLLLoss()
data_train = data.head(40000)

data_test = data.tail(2000)



X_train = data_train.drop('label', axis=1)

X_test = data_test.drop('label', axis=1)



y_train = data_train['label']

y_test = data_test['label']
import torch

epochs = 4000



data, target = torch.tensor(X_train.values, dtype=torch.float).cuda()/255, torch.tensor(y_train.values).cuda()

for epoch in range(epochs):

    optimizer.zero_grad()

    net_out = net(data)

    loss = criterion(net_out, target)

    loss.backward()

    optimizer.step()
net_out = net(data)

loss = criterion(net_out, target)

loss.data
data, target = torch.tensor(X_test.values, dtype=torch.float).cuda()/255, torch.tensor(y_test.values).cuda()

net_out = net(data)

test_loss = criterion(net_out, target).data

pred = net_out.data.max(1)[1]  # get the index of the max log-probability

correct = pred.eq(target.data).sum()



print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(

    test_loss, correct, len(data),

    100. * correct / len(data)))