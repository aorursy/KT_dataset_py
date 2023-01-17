# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
sample = pd.read_csv('/kaggle/input/1056lab-archit-heritage-elem-recognit/sampleSubmission.csv',index_col=0)
sample.head()
sample.info()
sample['class']=3
sample.head()
from os import path

from time import time

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import torch

import torch.nn as nn



import torchvision

import torchvision.transforms as transforms
use_cuda = torch.cuda.is_available()

print('Use CUDA:', use_cuda)
class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        self.l1 = nn.Linear(7*7*32, 1024)

        self.l2 = nn.Linear(1024, 1024)

        self.l3 = nn.Linear(1024, 10)

        self.act = nn.ReLU()

        self.pool = nn.MaxPool2d(2, 2)



    def __call__(self, x):

        h = self.pool(self.act(self.conv1(x)))

        h = self.pool(self.act(self.conv2(h)))

        h = h.view(h.size()[0], -1)

        h = self.act(self.l1(h))

        h = self.act(self.l2(h))

        h = self.l3(h)

        return h
model = CNN()

if use_cuda:

    model.cuda()



optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# ミニバッチサイズ・エポック数の設定

batch_size = 100

epoch_num = 50



# データセットの設定

train_data = FashionMNIST(root="./", train=True)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)



# 誤差関数の設定

criterion = nn.CrossEntropyLoss()

if use_cuda:

    criterion.cuda()



# ネットワークを学習モードへ変更

model.train()



# 学習の実行

loss_history=[]

for epoch in range(1, epoch_num+1):

    sum_loss = 0.0

    count = 0



    for image, label in train_loader:



        if use_cuda:

            image = image.cuda()

            label = label.cuda()



        y = model(image)



        loss = criterion(y, label)

        model.zero_grad()

        loss.backward()

        optimizer.step()



        sum_loss += loss.item()



        pred = torch.argmax(y, dim=1)

        count += torch.sum(pred == label)



    loss_history.append(sum_loss)

    print("epoch: {}, loss: {:.4f} accuracy: {:.4f}".format(epoch, sum_loss/(len(train_data)/batch_size), count.item()/len(train_data)))
# データローダーの準備

test_data = FashionMNIST(root="./", train=False)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False)



# ネットワークを評価モードへ変更

model.eval()



# 評価の実行

count = 0

all_preds = np.empty(0)

all_labels = np.empty(0)

with torch.no_grad():

    for image, label in test_loader:



        if use_cuda:

            image = image.cuda()

            label = label.cuda()

            

        y = model(image)



        pred = torch.argmax(y, dim=1)

        count += torch.sum(pred == label)

        

        p = pred.to("cpu").detach().numpy().copy()

        l = label.to("cpu").detach().numpy().copy()

        all_preds = np.append(all_preds, p)

        all_labels = np.append(all_labels, l)



print("test accuracy: {}".format(count.item() / len(test_data)))
f = open("result.csv", "w")

f.write("id,label\n")

for i, pred in enumerate(all_preds):

  f.write("{},{}\n".format(i+1, int(pred) ))

f.close()