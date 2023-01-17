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

import torch.nn.functional as F





class Net(nn.Module):



    def __init__(self):

        super(Net, self).__init__()

        # 输入图像channel：1；输出channel：6；5x5卷积核

        self.conv1 = nn.Conv2d(1, 6, 5) #2维卷积如何升维，用同一个二维卷积核？，即便用参数不同的卷积核有用吗？*其实因为连接了relu unit所以和同层多个神经元的效果一致

        self.conv2 = nn.Conv2d(6, 16, 5)

        # an affine operation: y = Wx + b

        self.fc1 = nn.Linear(16 * 5 * 5, 120)

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, 10)



    def forward(self, x):

        # 2x2 Max pooling

        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))

        # 如果是方阵,则可以只使用一个数字进行定义

        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = x.view(-1, self.num_flat_features(x))

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x



    def num_flat_features(self, x):

        size = x.size()[1:]  # 除去批处理维度的其他所有维度

        num_features = 1

        for s in size:

            num_features *= s

        return num_features





net = Net()

print(net)
params = list(net.parameters())

print(len(params))

print(params[0].size())#6个5*5卷积核
print(params[1].size())#6个对应卷积核的b
params[1]
print(params[2].size())#16个6*5*5的卷积核
params[3].size()# 16个6*5*5卷积核的b
params[4].size()#120个输入维度是16*5*5的神经元
params[5].size()#120个神经元的b
print(params[-1].size())#最后一个参数是b
input = torch.randn(1, 1, 32, 32)

out = net(input)

print(out)
input.size()
net.zero_grad()

out.backward(torch.randn(1, 10))#因为输出结果有十个数值
out.size()