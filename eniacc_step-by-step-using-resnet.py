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
import time

import torch

from torch import nn, optim

import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train=pd.read_csv("../input/digit-recognizer/train.csv")

test=pd.read_csv("../input/digit-recognizer/test.csv")

label=train["label"]

train=train.drop(labels = ["label"],axis = 1) 

train=train/255

test=test/255
train=train.values.reshape(-1,1,28,28) #1,28,28 单通道，28行 28列 # 由于使用图像增广，去掉一维

test=test.values.reshape(-1,1,28,28) #1,28,28 单通道，28行 28列
from keras.preprocessing.image import ImageDataGenerator

#  randomly rotating, scaling, and shifting

# CREATE MORE IMAGES VIA DATA AUGMENTATION

datagen = ImageDataGenerator(

        rotation_range=10,  

        zoom_range = 0.10,  

        width_shift_range=0.1, 

        height_shift_range=0.1

        )
train=torch.tensor(train, dtype=torch.float)

label=torch.tensor(label.values, dtype=torch.float)

batch_size = 32

train_data = torch.utils.data.TensorDataset(train,label)

train_iter = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True )
# 残差网络

class Residual(nn.Module): 

    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):

        super(Residual, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if use_1x1conv:

            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

        else:

            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.bn2 = nn.BatchNorm2d(out_channels)



    def forward(self, X):

        Y = F.relu(self.bn1(self.conv1(X)))

        Y = self.bn2(self.conv2(Y))

        if self.conv3:

            X = self.conv3(X)

        return F.relu(Y + X)
# 残差块

def resnet_block(in_channels, out_channels, num_residuals, first_block=False):

    if first_block:

        assert in_channels == out_channels # 第一个模块的通道数同输入通道数一致

    blk = []

    for i in range(num_residuals):

        if i == 0 and not first_block:

            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))

        else:

            blk.append(Residual(out_channels, out_channels))

    return nn.Sequential(*blk)
net = nn.Sequential(

    nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),

    nn.BatchNorm2d(32), 

    nn.ReLU(),

    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

    resnet_block(32, 32, 2, first_block=True),

    resnet_block(32, 64, 2),

    resnet_block(64, 128, 2),

    resnet_block(128, 256, 2),

    nn.AdaptiveAvgPool2d((1,1)),

    nn.Flatten(),

    nn.Linear(256,10),

)
lr=0.0015

decay=0

num_epochs=30

loss=nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=lr)
def train_net(net, train_iter,loss, num_epochs, lr, optimizer):

    for epochs in range(num_epochs):

        for x,y in train_iter:

            # 注意最后一次的数据不一定是符合batch_size的 所以用x.shape[0]代替

            if (epochs % 2)==0:

                b=np.array([datagen.flow(x[i:i+1]).next()[0] for i in range(x.shape[0])])

                x=torch.tensor(b, dtype=torch.float)

            y_hat=net(x) #模型计算

            l=loss(y_hat,y.long()) #损失计算

            optimizer.zero_grad() #梯度清0

            l.backward() #反向传播

            optimizer.step()

        print("epochs:"+str(epochs)+" loss:"+str(l))

        

train_net(net,train_iter,loss, num_epochs, lr, optimizer)
test=torch.tensor(test, dtype=torch.float)

sub=net(test)

sub=sub.argmax(dim=1)

sub=sub.numpy()

sub.reshape(-1,1)

submission=pd.read_csv("../input/digit-recognizer/sample_submission.csv")

submission["Label"]=sub

submission.to_csv('submission.csv', index=False)
submission