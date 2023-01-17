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
import torch

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader
class data(Dataset):

    def __init__(self, path):

        a = pd.read_csv(path)

        self.shape = a.shape[0]

        if a.shape[1] != 784:

            self.y_data = a['label']

            a = a.iloc[:,1:785] / 255

            self.x_data = a.values.reshape((-1, 1, 28, 28))

        else:

            self.y_data = None

            self.x_data = a.values.reshape((-1, 1, 28, 28))

    def __getitem__(self, index):

        if self.y_data is not None:

            return self.x_data[index], self.y_data[index]

        else:

            return self.x_data[index]

    def __len__(self):

        return self.shape

    

class model(nn.Module):

    def __init__(self):

        super(model, self).__init__()

        self.conv1 = nn.Conv2d(1, 128, 2, 1).cuda()

        self.batch1 = nn.BatchNorm2d(num_features=128).cuda()

        self.relu1 = nn.ReLU().cuda()

        self.max1 = nn.MaxPool2d(2,2).cuda()

        self.conv2 = nn.Conv2d(128, 128, 2, 1).cuda()

        self.batch2 = nn.BatchNorm2d(num_features=128).cuda()

        self.relu2 = nn.ReLU().cuda()

        self.max2 = nn.MaxPool2d(2,2).cuda()

        self.conv3 = nn.Conv2d(128, 128, 2, 1).cuda()

        self.batch3 = nn.BatchNorm2d(num_features=128).cuda()

        self.relu3 = nn.ReLU().cuda()

        self.module = nn.Sequential(self.conv1, self.relu1, self.batch1, self.max1,

                                    self.conv2, self.relu2, self.batch2, self.max2,

                                    self.conv3, self.relu3, self.batch3)

        fc1 = nn.Linear(3200, 256).cuda()

        fc2 = nn.Linear(256, 10).cuda()

        self.module2 = nn.Sequential(fc1, fc2)

        

    def forward(self, x):

        out = self.module(x)

        out = out.view(out.size(0), -1)

        out = self.module2(out)

        return out

    



    
c = model()

optimizer = optim.Adam(c.parameters(), 0.00001)

dat = data("../input/train.csv")

dat_test = data('../input/test.csv')

loader = DataLoader(dat, batch_size = 512, shuffle = True)

test = DataLoader(dat_test, batch_size = 512, shuffle = False)
#c.train()

device = torch.device('cuda:0')

for i in range(200):

    for x, y in loader:

        x = x.float()

        xs, ys = Variable(x), Variable(y)

        xs, ys = xs.cuda(), ys.cuda()

        optimizer.zero_grad()

        output = c(xs)

        loss = F.cross_entropy(output, ys)

        loss.backward()

        optimizer.step()

        

        
c.eval()

test_pred = torch.LongTensor()

for x in test:

    x = x.float()

    xs = Variable(x)

    xs = xs.cuda()

    output = c(xs)

    pred = output.cpu().data.max(1, keepdim=True)[1]

    test_pred = torch.cat((test_pred, pred), dim = 0)

    

test_pred.size()

out_df = pd.DataFrame(np.c_[np.arange(1, 28001)[:,None], test_pred.numpy()], columns=['ImageId', 'Label'])

out_df.to_csv('submission.csv', index=False)