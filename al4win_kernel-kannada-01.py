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

from torch.autograd import Variable

import torch.nn.functional as F

import torch.utils.data as Data

import torchvision
path = '/kaggle/input/Kannada-MNIST/'

data = pd.read_csv(path+ "train.csv")

train_y = data.values[:, 0]

train_x = data.values[:, 1:]

test = pd.read_csv(path +"test.csv")

test_x = test.values[:, 1:]





train_x = train_x / 255.0

train_x  = train_x .reshape(-1, 1, 28, 28)



test_x = test_x / 255.0 

test_x = test_x.reshape(-1, 1, 28, 28)
EPOCH = 5

BATCH_SIZE = 64

LR = 0.001
class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()

        

        self.conv1 = nn.Conv2d(1, 20, 5)

        self.conv2 = nn.Conv2d(20, 50, 5)

        self.fc1 = nn.Linear(4*4*50, 500)

        self.fc2 = nn.Linear(500, 10)

        

    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv2(x))

        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, 4*4*50)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

    

model = CNN()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # optimize all cnn parameters

loss_func = nn.NLLLoss() 

train_data = Data.TensorDataset(torch.FloatTensor(train_x), torch.LongTensor(train_y))

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)



# training...

for epoch in range(EPOCH):

    for step, (bx,by) in enumerate(train_loader) :

        

        output = model(bx)

        loss = loss_func(output, by)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        

        if step % 100 == 99 :

            pred_by = torch.max(output, 1)[1].data.squeeze()

            accuracy = (pred_by == by).sum().item() / float(BATCH_SIZE)

            print('Epoch:', epoch, '| train loss: %.3f' % loss.item(), '| train acc: % .2f' % accuracy)

            

# predicting...

test_data = Variable(torch.FloatTensor(test_x))

output = model(test_data)

predictions = torch.max(output, 1)[1].data.numpy().squeeze()
submission = pd.read_csv('/kaggle/input/Kannada-MNIST/sample_submission.csv')
submission['label'] = predictions
submission.to_csv("submission.csv",index=False)