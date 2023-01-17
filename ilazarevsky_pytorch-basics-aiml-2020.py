# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import torch

import torch.nn as nn

import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
a = torch.tensor([1,2,3,4.,6, 7])
b = torch.arange(6)
a = a.to("cuda:0")
a = torch.randn(5,2)
a.requires_grad_(True)
b = a * 2
print(b)
c = b.sum()
c
c.backward()
a.grad
a = torch.tensor(4., requires_grad=True)
b = a ** 2
b.backward()
a.grad
W = torch.randn(10, 4)
a = torch.randn(30, 10, dtype=torch.float32)
a.matmul(W) 
layer = nn.Linear(10,4)
torch.randn(1, 50) * torch.randn(50,1)
class Network():

    def __init__(self):

        hidden_size = 150

        self.W = torch.randn(1, hidden_size, requires_grad=True)

        self.b = torch.randn(hidden_size, requires_grad=True) # 50

        self.W2 = torch.randn(hidden_size,1, requires_grad=True)

        self.b2 = torch.randn(1,requires_grad=True)

        

    def forward(self, x):

        x = x.view(-1, 1) # B x 1 (batch size)

        h = x.matmul(self.W) # B x 50

        h = torch.tanh(h)

        h = h + self.b # B x 50

        y = h.matmul(self.W2) + self.b2 # B x 1

        return y.view(-1)

X_train = torch.rand(400) * 20 - 10
net = Network()
y_train = torch.sin(X_train)
learning_rate = 0.00002
iterator = tqdm(range(15000))



for i in iterator:

    y_pred = net.forward(X_train)

    loss = ((y_train - y_pred) ** 2).mean()

    loss.backward()

    

#     print(loss.item())

    net.W.data -= learning_rate * net.W.grad

    net.b.data -= learning_rate * net.b.grad

    net.W2.data -= learning_rate * net.W2.grad

    net.b2.data -= learning_rate * net.b2.grad

    

    net.W.grad, net.b.grad, net.W2.grad, net.b2.grad  = [None] * 4

    

    iterator.set_postfix({'loss': loss.item()})
X_test = torch.rand(100) * 20 - 10
with torch.no_grad():

    y_test = net.forward(X_test)
print(y_test)
sorting = torch.argsort(X_test)
torch.tensor([1,4,8,10])[[2,1]]
y_test[sorting]
plt.scatter(X_test.numpy(), np.sin(X_test.numpy()))

plt.plot(X_test[sorting].numpy(), y_test[sorting].numpy())
class Network(nn.Module):

    def __init__(self):

        super().__init__()

        hidden_size = 150

        self.linear1 = nn.Linear(1, hidden_size)

        self.linear2 = nn.Linear(hidden_size, 1)

    

        

    def forward(self, x):

        x = x.view(-1, 1) # B x 1 (batch size)

        h = self.linear1(x)

        h = torch.tanh(h)

        y = self.linear2(h)

        return y.view(-1)
net = Network()

net.cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
import os

os.system('nvidia-smi')
iterator = tqdm(range(5000))

X_train = X_train.cuda()

y_train = y_train.cuda()

for i in iterator:

    optimizer.zero_grad()

    y_pred = net(X_train)

    loss = ((y_train - y_pred) ** 2).mean()

    loss.backward()

    

    optimizer.step()

    iterator.set_postfix({'loss': loss.item()})

    
X_test = torch.rand(200) * 40 - 20

with torch.no_grad():

    y_test = net.forward(X_test.cuda()).cpu()

sorting = torch.argsort(X_test)
plt.scatter(X_test.numpy(), np.sin(X_test.numpy()))

plt.plot(X_test[sorting].numpy(), y_test[sorting].numpy())