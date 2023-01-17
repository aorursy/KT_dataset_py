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
## Mnist 手写体 ACC: 0.9948


import os 

import numpy as np

import time

import librosa

from IPython.display import Audio



import torch

import torchvision

import torchvision.transforms as transforms

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim
import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split



## 准备数据

test = pd.read_csv("../input/digit-recognizer/test.csv")

train = pd.read_csv("../input/digit-recognizer/train.csv")



train_src = np.array(train.iloc[:,1:])

train_label = np.array(train.iloc[:,:1])

test_src = np.array(test)





# print(train_src.shape)

# print(train_label.shape)

# print(test_src.shape)



# train_src = train_src  / 255

# test_src = test_src / 255

train_label = np.squeeze(train_label)



## 

X_train, X_test, Y_train, Y_test = train_test_split(train_src, train_label, test_size=0.1,random_state=10)



print("训练集：",X_train.shape,Y_train.shape)

print("验证集：",X_test.shape,Y_test.shape)

print(train_src[1])
## 建立数据加载器

from torch.utils.data import DataLoader,Dataset

import librosa



class myDataset(Dataset):

    def __init__(self,src,label,transform,isTrain):

        self.src = src.reshape(-1,28,28).astype(np.uint8)

        self.label = label

        self.transform = transform 

        self.isTrain = isTrain



    def __len__(self):

        return len(self.src)



    def __getitem__(self,index):

        if self.isTrain:

            return self.transform(self.src[index]),self.label[index]

        else:

            return self.transform(self.src[index])



# 并没有采用数据增强      

train_transform= transforms.Compose([transforms.ToPILImage(), \

                             transforms.RandomAffine(degrees=10),\

                             transforms.ToTensor(),\

                               transforms.Normalize(mean=(0.5,), std=(0.5,))])

test_transform = transforms.Compose([transforms.ToPILImage(), \

                             transforms.ToTensor(),\

                               transforms.Normalize(mean=(0.5,), std=(0.5,))])





train_dataset = myDataset(X_train,Y_train,train_transform,True)

valid_dataset = myDataset(X_test,Y_test,test_transform,True)

test_dataset = myDataset(test_src,None,test_transform,False)



train_dataloader = DataLoader(dataset=train_dataset,batch_size=256,shuffle=True)

valid_dataloader = DataLoader(dataset=valid_dataset,batch_size=256)

test_dataloader = DataLoader(dataset=test_dataset,batch_size=64)



for i,data in enumerate(valid_dataloader):

    print(i,data[0].shape,data[1].shape)
class ResidualBlock(nn.Module):

    def __init__(self, inchannel, outchannel, stride=1):

        super(ResidualBlock, self).__init__()

        self.left = nn.Sequential(

            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),

            nn.BatchNorm2d(outchannel),

            nn.ReLU(inplace=True),

            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),

            nn.BatchNorm2d(outchannel)

        )

        self.shortcut = nn.Sequential()

        if stride != 1 or inchannel != outchannel:

            self.shortcut = nn.Sequential(

                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),

                nn.BatchNorm2d(outchannel)

            )



    def forward(self, x):

        out = self.left(x)

        out += self.shortcut(x)

        out = F.relu(out)

        return out



class ResNet(nn.Module):

    def __init__(self, ResidualBlock, num_classes=10):

        super(ResNet, self).__init__()

        self.inchannel = 64

        self.conv1 = nn.Sequential(

            nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1,1), padding=1, bias=False),

            nn.BatchNorm2d(64),

            nn.ReLU(),

        )

        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)

        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)

        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)

        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)

        self.fc = nn.Linear(512, num_classes)

        self.softmax = nn.LogSoftmax()



    def make_layer(self, block, channels, num_blocks, stride):

        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]

        layers = []

        for stride in strides:

            layers.append(block(self.inchannel, channels, stride))

            self.inchannel = channels

        return nn.Sequential(*layers)



    def forward(self, x):

        x = x.view(-1,1,28,28)

        out = self.conv1(x)

        out = self.layer1(out)

        out = self.layer2(out)

        out = self.layer3(out)

        out = self.layer4(out)

        out = F.avg_pool2d(out, (3,3))

        out = out.view(out.size(0), -1)

        out = self.fc(out)

        out = self.softmax(out)

        return out





def ResNet18():

    return ResNet(ResidualBlock)
net = ResNet18().cuda()

net.train()
loss_fn = nn.NLLLoss()

optimizer = torch.optim.Adam(net.parameters(),lr=1e-4)

Epoches = 30
acc = 0

patience = 5

patience_count = 0

for epoch in range(Epoches):

    if patience_count >= patience: ## 提前结束的条件

        break

    for i,data in enumerate(train_dataloader):

        src = data[0].type(torch.FloatTensor).cuda()

        label = data[1].type(torch.LongTensor).cuda()

        output = net(src)

        loss = loss_fn(output, label)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        

        if i % 10 == 0:

            print("第%d个ecoch中，第%d个batch的损失函数为：%f" % (epoch,i,loss.cpu().detach().numpy()))

    

    loss_avg = []

    acc_avg = []

    for i,data in enumerate(valid_dataloader):

        src = data[0].type(torch.FloatTensor).cuda()

        label = data[1].type(torch.LongTensor).cuda()

        output = net(src)

        loss = loss_fn(output, label)

        loss_avg.append(loss.cpu().detach().numpy())



        pre = torch.argmax(output,dim=1).cpu()

        s = (label.cpu() == pre).sum()

        acc_avg.append(s.numpy() / len(label) )



    print("########################################")

    print("Loss为：",np.mean(loss_avg))

    print("acc_avg的正确率为：",np.mean(acc_avg))

    if acc < np.mean(acc_avg):

        print("超过历史最高正确率",acc,np.mean(acc_avg))

        acc = np.mean(acc_avg)

        torch.save(net, './model_2.pkl')

        patience_count = 0

    else:

        patience_count += 1

    print("#######################################")
## 获取结果

net = torch.load('./model_2.pkl')

net.eval()

pre_test = []

loss_avg = []

for i,data in enumerate(valid_dataloader):

    src = data[0].type(torch.FloatTensor).cuda()

    print(src.shape)

    output = net(src)

    pre = torch.argmax(output,dim=1).cpu()

    pre_test.extend(pre.data.numpy())



pre_test = np.array(pre_test)

print(pre_test.shape)
## 查看验证集的正确率

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import confusion_matrix



fig = plt.figure(figsize=(10, 10)) # Set Figure

mat = confusion_matrix(Y_test, pre_test) # Confusion matrix

# Plot Confusion matrix

sns.heatmap(mat.T, square=True, annot=True, cbar=False, cmap=plt.cm.Blues)

plt.xlabel('Predicted Values')

plt.ylabel('True Values');

plt.show();



## 预测并提交结果

net = torch.load('./model_2.pkl')

net.eval()

pre_test = []

loss_avg = []

for i,data in enumerate(test_dataloader):

    src = data.type(torch.FloatTensor).cuda()

    print(src.shape)

    output = net(src)

    pre = torch.argmax(output,dim=1).cpu()

    pre_test.extend(pre.data.numpy())

    

sub = pd.read_csv('../input/digit-recognizer/sample_submission.csv')

sub['Label'] = pre_test

sub.to_csv("CNN_sub.csv", index=False)

sub.head()
