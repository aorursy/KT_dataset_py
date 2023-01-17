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
# モジュールのインポート

from time import time

import torch

import torch.nn as nn



import torchvision

import torchvision.transforms as transforms



# GPUの確認

use_cuda = torch.cuda.is_available()

print('Use CUDA:', use_cuda)

import glob
root = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']



#X_train = glob.glob('/kaggle/input/1056lab-archit-heritage-elem-recognit/train/'+str(root)+'/*')

X_train = glob.glob('/kaggle/input/1056lab-archit-heritage-elem-recognit/train/*/*')

Y_train = root

X_test = glob.glob('/kaggle/input/1056lab-archit-heritage-elem-recognit/test/*')



print(X_train)

print(X_test)

print(Y_train)
class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        self.l1 = nn.Linear(7 * 7 * 32, 1024)

        self.l2 = nn.Linear(1024, 1024)

        self.l3 = nn.Linear(1024, 10)

        self.act = nn.ReLU()

        self.pool = nn.MaxPool2d(2, 2)

    

    def forward(self, x):

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

batch_size = 64

epoch_num = 10

n_iter = len(X_train) / batch_size



# データローダーの設定

train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True)



# 誤差関数の設定

criterion = nn.CrossEntropyLoss()

if use_cuda:

    criterion.cuda()



# ネットワークを学習モードへ変更

model.train()



start = time()

for epoch in range(1, epoch_num+1):

    sum_loss = 0.0

    count = 0

    

    for image ,label in train_loader:

        

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

        

    print("epoch: {}, mean loss: {}, mean accuracy: {}, elapsed_time :{}".format(epoch,

                                                                                 sum_loss / n_iter,

                                                                                 count.item() / len(train_loader),

                                                                                 time() - start))
submit_df = pd.read_csv('/kaggle/input/1056lab-archit-heritage-elem-recognit/sampleSubmission.csv', index_col=0)

submit_df['class'] = 0

submit_df
submit_df.to_csv('submission.csv')