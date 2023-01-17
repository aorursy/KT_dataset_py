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
path_to_train = "../input/fruits-360_dataset/fruits-360/Training/"

path_to_test = "../input/fruits-360_dataset/fruits-360/Test/"

classes = os.listdir(path_to_train)

value_of_img = 0

for ii in classes:

    value_of_img += len(os.listdir(path_to_train+ii))

print(value_of_img)







# read the image

classes_lib = path_to_train + classes[0]

ims = os.listdir(classes_lib)

im = plt.imread(classes_lib+"/"+ims[0])

# show the image

print(im.shape)

plt.imshow(im)

plt.show()



classes.sort()
def new_train_data(a): 

    x_train = torch.zeros(a,100,100,3)

    y_train = torch.zeros(a).type(torch.LongTensor)

    for ii in torch.arange(a):

        rndcls = torch.randint(0,len(classes),(1,))[0]

        clsdir = os.listdir(path_to_train + classes[rndcls])

        rndim = plt.imread(path_to_train + classes[rndcls] + "/" + clsdir[torch.randint(0,len(clsdir),(1,))[0]])

        y_train[ii] = rndcls

        x_train[ii] = torch.tensor(rndim)

    return x_train.transpose(1,3)/255, y_train

def new_valid_data(a):  

    x_test = torch.zeros(a,100,100,3)

    y_test = torch.zeros(a).type(torch.LongTensor)

    for ii in torch.arange(a):

        rndcls = torch.randint(0,len(classes),(1,))[0]

        clsdir = os.listdir(path_to_test + classes[rndcls])

        rndim = plt.imread(path_to_test + classes[rndcls] + "/" + clsdir[torch.randint(5,len(clsdir),(1,))[0]])

        y_test[ii] = rndcls

        x_test[ii] = torch.tensor(rndim)

    return x_test.transpose(1,3)/255, y_test

def test_data():

    x_test = torch.zeros(len(classes)*5,100,100,3)

    y_test = torch.zeros(len(classes)*5).type(torch.LongTensor)

    for ii in range(len(classes)):

        clsdir = os.listdir(path_to_test + classes[ii])

        for jj in range(5):

            im = plt.imread(path_to_test + classes[ii] + "/" + clsdir[jj])

            y_test[ii*5+jj] = ii

            x_test[ii*5+jj] = torch.tensor(im)

    return x_test.transpose(1,3)/255, y_test
import torch.nn as nn

import torch.nn.functional as F





class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 30, 3)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(30, 90, 3)

        self.fc1 = nn.Linear(90*23*23, 500)

        self.fc2 = nn.Linear(500, len(classes))

        #self.fc3 = nn.Linear(84, len(classes))



    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))

        #print("shape after conv1+pool",x.shape)

        x = self.pool(F.relu(self.conv2(x)))

        #print("shape after conv2+pool",x.shape)

        x = x.view(-1, 90*23*23)

        #print("shape after reshape",x.shape)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)#F.softmax(self.fc2(x), dim=1)

        #print(x)

        #print("shape after linear2",x.shape)

        #x = self.fc3(x)

        #print("shape after linear3",x.shape)

        return x

device = torch.device("cuda")

net = Net()

net.to(device)
import torch.optim as optim



criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
best = net

best_loss = 1000.0

for epoch in range(10):

    print("epoch:",epoch+1)

    for i in range(250):

        # get the inputs

        inputs, labels = new_train_data(200)

        inputs, labels = inputs.to(device), labels.to(device)



        optimizer.zero_grad()



        # forward + backward + optimize

        outputs = net(inputs)

        loss = criterion(outputs, labels)

        if loss < best_loss:

            print(loss)

            best = net

            best_loss = loss

        loss.backward()

        optimizer.step()



print('Finished Training')

#Валидация

vx, vy = new_valid_data(1000)

vx = vx.to(device)

output = best(vx)

pred = output.max(1)[1]

print(classification_report(vy, pred.cpu(), labels=torch.arange(len(classes))))
xx, yy = test_data()

xx = xx.to(device)

output = best(xx)

pred = output.max(1)[1]

print(classification_report(yy, pred.cpu(), labels=torch.arange(len(classes))))

print(confusion_matrix(yy, pred.cpu(), labels=torch.arange(len(classes))))