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
# Code from MorvanZhou/PyTorch-Tutorial



import torch

import torch.nn as nn

from torch.autograd import Variable

import torch.utils.data as Data

import torchvision

import time



torch.manual_seed(1)



import matplotlib.pyplot as plt

%matplotlib inline



EPOCH = 1

BATCH_SIZE = 50

LR = 0.001

DOWNLOAD_MNIST = True #False



train_data = torchvision.datasets.MNIST(

    root='./mnist/', 

    train=True,

    transform=torchvision.transforms.ToTensor(), 

    download=DOWNLOAD_MNIST,)



train_loader = Data.DataLoader(

    dataset=train_data, 

    batch_size=BATCH_SIZE, 

    shuffle=True)



test_data = torchvision.datasets.MNIST(

    root='./mnist/', train=False)



# !!!!!!!! Change in here !!!!!!!!! #

test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1)).type(torch.FloatTensor)[:2000].cuda()/255.   # Tensor on GPU

test_y = test_data.test_labels[:2000].cuda()



class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2,),                      

            nn.ReLU(), 

            nn.MaxPool2d(kernel_size=2),)

        self.conv2 = nn.Sequential(

            nn.Conv2d(16, 32, 5, 1, 2), 

            nn.ReLU(), 

            nn.MaxPool2d(2),)

        self.out = nn.Linear(32 * 7 * 7, 10)



    def forward(self, x):

        x = self.conv1(x)

        x = self.conv2(x)

        x = x.view(x.size(0), -1)

        output = self.out(x)

        return output



cnn = CNN()



# !!!!!!!! Change in here !!!!!!!!! #

cnn.cuda()      # Moves all model parameters and buffers to the GPU.



optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)

loss_func = nn.CrossEntropyLoss()



#losses_his = []

#Training

t1 = time.time()

for epoch in range(EPOCH):

    for step, (x, y) in enumerate(train_loader):



        # !!!!!!!! Change in here !!!!!!!!! #

        b_x = Variable(x).cuda()    # Tensor on GPU

        b_y = Variable(y).cuda()    # Tensor on GPU

        #print("b_y", b_y.shape, b_y)



        output = cnn(b_x)

        loss = loss_func(output, b_y)

        #losses_his.append(loss.data[0])

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()



        if step % 50 == 0:

            test_output = cnn(test_x)



            # !!!!!!!! Change in here !!!!!!!!! #

            pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()  # move the computation in GPU



            accuracy = torch.sum(pred_y == test_y).type(torch.FloatTensor) / test_y.size(0)

            print('Epoch: ', epoch, '| train loss: %.4f' % loss, '| test accuracy: %.2f' % accuracy)

            

t2 = time.time()

print("runtime:", t2- t1)







import scipy.ndimage

from scipy.ndimage import gaussian_filter

from scipy.ndimage import rotate



test_output = cnn(test_x[:10])

pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()

for i in range(10):

    img = test_data.data[i]

    valid_angles = 0 #np.random.uniform(0.0, 359.9)#[0, 90, 180, 270]

    ang = valid_angles #np.random.choice(valid_angles)

    #print("ang", ang)

    img = rotate(img, ang, reshape=False)



    plt.imshow(img)

    pre = pred_y[np.int(i)]

    print("neural network prediction for image below:", pre)

    #print("ang", ang)

    #plt.title("neural network prediction:", pre)

    plt.show()
#### below for testing rotational effect https://arxiv.org/pdf/1712.02779.pdf





rotate_output = cnn(torch.from_numpy(img).float().cuda().unsqueeze(0).unsqueeze(0)) 
plt.imshow(img)

plt.show()

rotate_output

pred_y = torch.max(rotate_output, 1)[1].cuda().data.squeeze()

pred_y
print(test_x[0].shape)
for i in range(10):

    img = test_data.data[i]

    valid_angles = 0 #valid_angles = np.random.uniform(-20.0, 20.9)#[0, 90, 180, 270]

    ang = valid_angles #np.random.choice(valid_angles)

    #print("ang", ang)

    img = rotate(img, ang, reshape=False)



    rotate_output = cnn(torch.from_numpy(img).float().cuda().unsqueeze(0).unsqueeze(0)) 

    pred_y = torch.max(rotate_output, 1)[1].cuda().data.squeeze()



    

    plt.imshow(img)

    

    #pre = pred_y[np.int(i)]

    print("neural network prediction for image below:", pred_y)

    print("ang", ang)

    #plt.title("neural network prediction:", pre)

    plt.show()