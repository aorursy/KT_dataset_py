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
import os

import time

import torch

from torch import optim

import torch.nn as nn

import numpy as np

from PIL import Image

from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from torchvision import datasets

import torch.nn.functional as F
def TPCA(k, imgs):

    imgIdx = 0     # 图片的下标

    imgNum = len(imgs)

    X = np.zeros((k*k, imgNum*128*128))

    m,n = 128,128

    imgCount = 0

    lk = (k-1)//2

    rk = (k+1)//2

    for img in imgs:

        img = transforms.ToTensor()(img)*255    # PIL.Image -> Tensor

        img = img.squeeze()

        # 对一张图片得到每个patch的Vector向量

        # 加padding，然后再取patch的切片，然后拉成向量

        zeroP = torch.nn.ZeroPad2d(lk)

        img = zeroP(img)

        x = np.zeros((k*k, 128*128))

        vectorNum = 0

        for i in range(lk, m + lk):    

            for j in range(lk, n + lk):

                nowVector = img[i-lk:i+rk, j-lk:j+rk].clone()

                nowVector = nowVector.view(-1)

                x[:, vectorNum] = np.array(nowVector).T

                vectorNum += 1

        X[:, imgCount*m*n:(imgCount+1)*m*n] = x

        imgCount += 1

    X = X-X.mean(0)   # 去均值化

    print(X)

    R = X.dot(X.T)

    W = np.linalg.eig(R)

    print('shape:', W[0].shape, W[1].shape)  

    L1 = 8

    WL = W[1][:,:L1].T  # 得到前L1个特征值。

    WL = WL.reshape(-1, k, k)  # 8*7*7

    return WL
# Encode

class Encode(nn.Module):

  # 需要提供imgs来初始化。

    def __init__(self):

        super(Encode, self).__init__()

        # 所有图片根据PCANet提取的特征：

        self.Conv1 = nn.Sequential(

            # in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=1

            nn.Conv2d(1, 16, 3, 1, 1),     # 128*128*16

            nn.ReLU(),

            nn.MaxPool2d(2)            # 64*64*16

        )

#         self.Conv2 = nn.Sequential(

#             nn.Conv2d(16, 32, 3, 1, 1),    # 64*64*32

#             nn.ReLU(),

#             nn.MaxPool2d(2)            # 32*32*32

#         )

        self.Linear = nn.Linear(64*64*16,1024)   # 32*32*32 -> 1024



    def forward(self, inputImg):

        x = self.Conv1(inputImg)

#         x = self.Conv2(x)

        x = x.view(x.size(0), -1)   # num * (32*32*32)



        return self.Linear(x)       # 1024
# Decode

class Decode(nn.Module):

    def __init__(self,indim): # indim = 1024

        nn.Module.__init__(self)

        self.Linear1 = nn.Linear(indim,128*128*1) 

#         self.Conv1 = nn.Sequential(

#             nn.Conv2d(32,16,3,1,2),  # 112*112*32 -> 112*112*16

#             nn.Relu(),

#             nn.MaxPool2d(2)  # 56*56*16

#         )

        self.Conv2 = nn.Sequential(  

            nn.Conv2d(1,3,3,1,1),  # 128*128*1 -> 128*128*3

            nn.ReLU(),

#             nn.MaxPool2d(2)   # 28*28*1

        )

    def forward(self,x): 

        x = self.Linear1(x)

        x = x.view(-1, 1, 128, 128)

        # x = self.Conv1(x)

        x = self.Conv2(x)

        return x

   
class GModel(nn.Module):

    def __init__(self, encode, decode):

        super(GModel, self).__init__()

        self.encode = encode

        self.decode = decode



    def forward(self, x):

        x = self.encode(x)

        return self.decode(x)
def dealImg(inputImg):

    """input:PIL.Image

       output:PIL.Image

       作用：对图像进行PCANet处理

    """

    L1 = 8

    Ii = [0]*L1

    m,n = inputImg.size

    YY = np.zeros((m,n))

#     YY = torch.zeros(1,1,m,n)     # 0张量

    inputImg = torch.unsqueeze(transforms.ToTensor()(inputImg), 0)    # PIL.Image(128*128*1) ->Tensor(1*1*128*128)

    for i in range(L1):

        tw = torch.Tensor(WL[np.newaxis, np.newaxis, i, :, :])   # 1*1*7*7 

        Ii[i] = F.conv2d(inputImg, tw, bias = None, stride = 1, padding = 3)

        Ii[i] = Ii[i].view(Ii[i].size(2), Ii[i].size(3))

        meani = Ii[i].mean()

        # 二值化*权值

        Ii[i][Ii[i] < meani] = 0

        Ii[i][Ii[i] >= meani] = 2**(7-L1) 

        YY += np.array(Ii[i])

    YY = np.asarray(YY, np.uint8) 

    YY = Image.fromarray(YY)         # numpy.array -> PIL.Image

    return YY
# WL = np.random.rand(8, 7, 7)

# WL = TPCA(7, imgs)

for img in imgs:

    inputImg = img

#     print(img.size)

    YY = dealImg(inputImg)

    YY = YY.squeeze()

#     print(YY.shape)

    YY = np.asarray(YY, np.uint8)

    Image.fromarray(YY)

    plt.subplot(121)

    plt.imshow(YY.squeeze())

    plt.subplot(122)

    plt.imshow(img)

    plt.show()
# 数据集

class MyDataSet(Dataset):

    def __init__(self, dataset, transform=None):

        self.dataset = dataset

        self.transform = transform

        self.image = [dealImg(data) for data in self.dataset]

        self.label = [data for data in self.dataset]

    

    def __len__(self):

        return len(self.dataset)

    

    def __getitem__(self, idx):

        image = self.image[idx]

        label = self.label[idx]

        if transform is not None:   # image 已经被转换成Tensor ，则不需要指定transform

            image = transform(image)  

            label = transform(label)    # 此处label也要进行transform

#         print(type(image))



        return image,label  # ,label
# 数据集准备

basePath = r'/kaggle/input/facedata/'

imgNames = os.listdir(basePath)

path = r'/kaggle/input'

epochs = 100

transform = transforms.Compose([

    transforms.ToTensor(),

])

dataset = []

for imgName in imgNames:

    img = Image.open(os.path.join(basePath, imgName)).convert('L')

    dataset.append(img)



mydataset = MyDataSet(dataset, transform)

print("train img num:",len(mydataset))     # 10691

mydataloader = DataLoader(mydataset, batch_size=128, shuffle=True)  # works



# PCA数据集准备(imgs)

imgs = []

count = 0

for imgName in imgNames:

    img = Image.open(os.path.join(basePath, imgName)).convert('L')

#     print(img, img.size)

    imgs.append(img)

    count += 1

    if count >= 600:

        break
# 模型初始化

criterion = nn.MSELoss()  # 均方误差

encoder = Encode()

decoder = Decode(1024)

gModel = GModel(encoder,decoder)

# 使用GPU

use_gpu = torch.cuda.is_available()

if use_gpu:

    gModel = gModel.cuda()

    criterion = criterion.cuda()

optimizer = torch.optim.Adam(gModel.parameters(), lr=0.003)
# 训练

for epoch in range(30):

    for i, data in enumerate(mydataloader):

        print(i)

        inputs, label = data

        inputs, label = Variable(inputs), Variable(label)

        optimizer.zero_grad()

        if use_gpu:

            inputs = inputs.cuda()

        dec = gModel(inputs)

        loss = criterion(dec, label.cuda())

        loss.backward()

        print(loss)

#         for par in gModel.parameters():

#             print(par.grad.size(), par.grad)

        optimizer.step()

        l = loss.item()

    if epoch % 1 == 0:

#         torch.save(vae.state_dict(), PATH)

        print(epoch, l)
PATH = 'gModel.pt'

torch.save(gModel.state_dict(), PATH)
testDataLoader = DataLoader(mydataset, batch_size=1, shuffle=False)  # works

import matplotlib.pyplot as plt

for i, data in enumerate(testDataLoader):

    if i == 2:

        inputs, label = data

        inputs, label = Variable(inputs), Variable(label)

        if use_gpu:

            inputs = inputs.cuda()

        output = gModel(inputs)

        loss = criterion(output, inputs)

        print(loss)

        plt.subplot(1, 2, 1)

    #     print(output.cpu().squeeze().size())

        outImg = transforms.ToPILImage()(output.cpu().squeeze())    # 1*3*128*128 Tensor -> 3*128*128 PIL.Image

        print(np.array(outImg))

    #     print(np.array(outImg).shape)

        plt.imshow(outImg)

        inImg = transforms.ToPILImage()(inputs.cpu().squeeze())

        print('inImg', np.array(inImg))

        plt.subplot(1, 2, 2)

        plt.imshow(inImg)

        plt.show()

    #     print(output.size())

    #     print(inputs.size())

        break
x = -0.618

print(x**4 + 2*x + 4)
np.__version__
mse = nn.MSELoss()

testx = torch.rand(128, 128)

testy = torch.rand(128, 128)



loss = mse(testx, testy)

print(loss)