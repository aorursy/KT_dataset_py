# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import torch

#import torchvision

import matplotlib.pyplot as plt

torch.manual_seed(0)

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

# Any results you write to the current directory are saved as output.

path_to_train = "../input/sign_data/sign_data/train/"

path_to_test = "../input/sign_data/sign_data/test/"

classes = os.listdir(path_to_train)



# read the image

classes_lib = path_to_train + classes[0]

ims = os.listdir(classes_lib)

im = plt.imread(classes_lib+"/"+ims[1])

# show the image

print(classes[0],im.shape)

value_of_test_img = len(classes)*3

plt.imshow(im)

plt.show()



#for i in range(im.shape[0]):

#    for j in range(im.shape[1]):

#        if im[i,j,0]+im[i,j,1]+im[i,j,2] < 2.75:

#            im[i,j] = (0,0,0)

#        else:

#            im[i,j] = (1,1,1)



#plt.imshow(im)

#plt.show()



classes.sort()

test_classes = classes[100:]

train_classes = classes[:100]



value_of_img = 0

for ii in train_classes:

    value_of_img += len(os.listdir(path_to_train+ii))

print(value_of_img)



value_of_imgs = 0

for ii in test_classes:

    value_of_imgs += len(os.listdir(path_to_train+ii))

print(value_of_imgs)



print(test_classes,train_classes)
from PIL import Image
x_train = torch.zeros(value_of_img,83,229,3)

y_train = torch.zeros(value_of_img).type(torch.LongTensor)

pointer = 0

for i in train_classes:

    imgs = os.listdir(path_to_train+i)#[4:]

    for j in imgs:

        imgg = Image.open(path_to_train+i+"/"+j)

        img = imgg.resize((229, 83))

        x_train[pointer] = torch.tensor(list(img.getdata())).reshape(83,229,3)

        if i.find('forg') != -1:

            y_train[pointer] = 0

        else:

            y_train[pointer] = 1

        pointer += 1

x_train = x_train/255
x_valid = torch.zeros(len(classes),83,229,3)

y_valid = torch.zeros(len(classes)).type(torch.LongTensor)

pointer = 0

for i in classes:

    imgs = os.listdir(path_to_train+i)[3:4]

    for j in imgs:

        imgg = Image.open(path_to_train+i+"/"+j)

        img = imgg.resize((229, 83))

        x_valid[pointer] = torch.tensor(list(img.getdata())).reshape(83,229,3)

        if i.find('forg') != -1:

            y_valid[pointer] = 0

        else:

            y_valid[pointer] = 1

        pointer += 1

x_valid = x_valid/255
x_test = torch.zeros(value_of_test_img,83,229,3)

y_test = torch.zeros(value_of_test_img).type(torch.LongTensor)

pointer = 0

for i in test_classes:

    imgs = os.listdir(path_to_train+i)#[0:3]

    for j in imgs:

        imgg = Image.open(path_to_train+i+"/"+j)

        img = imgg.resize((229, 83))

        x_test[pointer] = torch.tensor(list(img.getdata())).reshape(83,229,3)

        if i.find('forg') != -1:

            y_test[pointer] = 0

        else:

            y_test[pointer] = 1

        pointer += 1

x_test = x_test/255
print(y_test[44].item(),y_test.shape)

plt.imshow(x_test[44])

plt.show()
from torch.utils.data import TensorDataset,DataLoader
train_ds = TensorDataset(x_train, y_train)

valid_ds = TensorDataset(x_valid, y_valid)

test_ds = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_ds,batch_size=100, shuffle=True)

valid_loader = DataLoader(valid_ds,batch_size=len(classes))

test_loader = DataLoader(test_ds,batch_size=value_of_imgs)
import torch.nn as nn

import torch.nn.functional as F
class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 30, 3)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(30, 90, 3)

        self.fc1 = nn.Linear(90*55*19, 500)

        self.fc2 = nn.Linear(500, len(train_classes))



    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))

        #print("shape after conv1+pool",x.shape)

        x = self.pool(F.relu(self.conv2(x)))

        #print("shape after conv2+pool",x.shape)

        x = x.view(-1, 90*55*19)

        #print("shape after reshape",x.shape)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x

device = torch.device("cuda")

net = Net()

net.to(device)
import torch.optim as optim



criterion = nn.TripletMarginLoss()

optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

best = net

best_loss = 1000.0

anchor = torch.zeros(len(train_classes))

for epoch in range(11):

    print("epoch:",epoch+1)

    for inputs, labels in train_loader:

        inputs = inputs.transpose(1,3)

        inputs, labels = inputs.to(device), labels.to(device)



        optimizer.zero_grad()



        # forward + backward + optimize

        outputs = net(inputs)

        

        positive = torch.masked_select(outputs,labels.reshape(-1,1).type(torch.uint8)).reshape(-1,len(train_classes))

        negative = torch.masked_select(outputs,torch.where(labels == 0, torch.ones_like(labels), torch.zeros_like(labels)).reshape(-1,1).type(torch.uint8)).reshape(-1,len(train_classes))

        less = 0

        if positive.shape[0] < negative.shape[0]:

            less = positive.shape[0]

        else:

            less = negative.shape[0]

        loss = criterion(positive[0], positive[:less], negative[:less]) #(anchor, positive, negative)

        if loss < best_loss:

            print(loss)

            #print(positive[0])

            anchor = positive[0]

            best = net

            best_loss = loss

        loss.backward()

        optimizer.step()



print('Finished Training')
print(anchor)
#Валидация

#for vx, vy in valid_loader:

#    vx = vx.to(device).transpose(1,3)

#    output = best(vx)

#    pred = output.max(1)[1]

#    print(classification_report(vy, pred.cpu(), labels=torch.arange(2)))
pdist = nn.PairwiseDistance(p=2)

for xx, yy in test_loader:

    xx = xx.to(device).transpose(1,3)

    output = best(xx)

    predt = pdist(output.cpu(),anchor.cpu())

    print(predt)

    print(predt.sum()/len(predt))

    disp = (predt-predt.sum()/len(predt)).pow(2).sum()/len(predt)

    norm_disp = (disp).pow(1/2)/2

    print(norm_disp)

    

    pred = torch.where(predt < predt.sum()/len(predt) - norm_disp, torch.ones_like(yy), torch.zeros_like(yy))

    print(classification_report(yy, pred.cpu(), labels=torch.arange(2)))

    print(confusion_matrix(yy, pred.cpu(), labels=torch.arange(2)))

    break;