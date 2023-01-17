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

import torchvision

from torchvision import transforms, datasets

import torch.nn as nn

import torch.nn.functional as F

import numpy as np

import matplotlib.pyplot as plt

import tqdm

from numpy import genfromtxt
train_data = genfromtxt('/kaggle/input/Kannada-MNIST/train.csv', delimiter=',')
test_data = genfromtxt('/kaggle/input/Kannada-MNIST/test.csv', delimiter=',')
train_data = train_data[1:]

test_data =test_data[1:]

train = []

test = []
for i in range(len(train_data)):

    train.append((torch.tensor(train_data[i][1:], dtype=torch.float64).reshape(1,28,28) , int(train_data[i][0])))
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
class Net(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(28*28, 64)

        self.fc2 = nn.Linear(64, 64)

        self.fc3 = nn.Linear(64, 64)

        self.fc4 = nn.Linear(64, 10)



    def forward(self, x):

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))

        x = self.fc4(x)

        return F.log_softmax(x, dim=1)



net = Net()

print(net)
import torch.optim as optim
loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=0.001)
for epoch in range(10): # 3 full passes over the data

    for data in trainset:  # `data` is a batch of data

        X, y = data  # X is the batch of features, y is the batch of targets.

        net.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.

        output = net(X.float().view(-1,784))  # pass in the reshaped batch (recall they are 28x28 atm)

        loss = F.nll_loss(output, y)  # calc and grab the loss value

        loss.backward()  # apply this loss backwards thru the network's parameters

        optimizer.step()  # attempt to optimize weights to account for loss/gradients

    print(loss)  # print loss. We hope loss (a measure of wrong-ness) declines! 
for i in range(len(test_data)):

    test.append((torch.tensor(test_data[i][1:], dtype=torch.float64).reshape(1,28,28) , int(test_data[i][0])))
with torch.no_grad():

    for data in test[:100]:

        X, y = data

        output = net(X.float().view(-1,784))

        #print(output)

        for i in output:

            print(y, ':', int(torch.argmax(i)))
# write file

f = open('output.csv', 'w+')

f.write('id,label\n')



with torch.no_grad():

    for data in test:

        X, y = data

        output = net(X.float().view(-1,784))

        #print(output)

        for i in output:

            #print(y, ':', int(torch.argmax(i)))

            txt = '{},{}\n'.format(int(y), int(torch.argmax(i)))

            f.write(txt)

            

f.close()