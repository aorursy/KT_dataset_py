# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/train.csv")
import torch

import torch.nn as nn

import torch.utils.data as data

import torchvision.datasets as dsets

import torchvision.transforms as transforms

from torch.autograd import Variable
class Mnist(data.Dataset):

    def __init__(self):

        train_X = df[0:20000]

        train_Y = train_X.label.as_matrix().tolist()

        train_X = train_X.drop("label",axis=1).as_matrix().reshape(20000,1,28,28)

        self.datalist = train_X

        self.labellist = train_Y





    def __getitem__(self, index):

        return torch.Tensor(self.datalist[index].astype(float)), self.labellist[index]



    def __len__(self):

        return self.datalist.shape[0]



train_data = Mnist()

train_loader = torch.utils.data.DataLoader(dataset=train_data,

                                           batch_size=50, 

                                           shuffle=True,

                                           num_workers=2)
class CNN(nn.Module):

    def __init__(self):

        super(CNN,self).__init__()

        self.layer1 = nn.Sequential(

            nn.Conv2d(1, 20, kernel_size=5),

            nn.BatchNorm2d(20),

            nn.ReLU(),

            nn.MaxPool2d(2))

        self.fc = nn.Linear(12*12*20, 10)





    def forward(self, x):

        out = self.layer1(x)

        out = out.view(out.size(0),-1)

        out = self.fc(out)

        return out

    

cnn = CNN()
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(cnn.parameters(), lr=0.001)
for epoch in range(5):

    for i, (images, labels) in enumerate(train_loader):

        images = Variable(images)

        labels = Variable(labels)



        optimizer.zero_grad()

        outputs = cnn(images)



        loss = loss_fn(outputs, labels)

        loss.backward()

        optimizer.step()



        if (i+1) % 100 == 0:

            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' %(epoch+1, 5, i+1, len(train_data)//50,  loss.data[0]))

testX = df[20000:22000]

testY = testX['label'].values

testX = testX.drop('label',axis=1).as_matrix().reshape(2000,1,28,28).astype(float)
testX = Variable(torch.Tensor(testX))

pred = cnn(testX)

_, predlabel = torch.max(pred.data, 1)
np.sum(predlabel.numpy()==testY)/2000
from sklearn.metrics import classification_report

print(classification_report(testY,predlabel))