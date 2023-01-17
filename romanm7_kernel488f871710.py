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

import torchvision

from torch.utils.data import TensorDataset,DataLoader

from torchvision import transforms
comp = transforms.Compose([

        transforms.Resize(size=(150,150)),

        transforms.ToTensor()])



train_ds = torchvision.datasets.ImageFolder("../input/seg_train/seg_train/", transform=comp)

train_loader = DataLoader(train_ds,batch_size=128, shuffle=True)

val_dataset = torchvision.datasets.ImageFolder("../input/seg_test/seg_test/", transform=comp)

val_ds, test_ds = torch.utils.data.random_split(val_dataset, [1500, 1500])

val_loader = DataLoader(val_ds, batch_size=128)

test_loader = DataLoader(test_ds, batch_size=128)
import torch.nn as nn

import torch.nn.functional as F

import math

class Model(nn.Module):

    def __init__(self):

        super(Model, self).__init__()

        self.properties = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=75, kernel_size=3, padding=2),

            nn.ReLU(),

            nn.Conv2d(in_channels=75, out_channels=100, kernel_size=3),

            nn.ReLU(),

            nn.MaxPool2d(10),

            nn.ReLU(),

            nn.Conv2d(in_channels=100, out_channels=120, kernel_size=3),

            nn.MaxPool2d(3)

        )

            

        self.estimator = nn.Sequential(

            nn.Linear(1920,500),

            nn.ReLU( inplace=True ),

            nn.Linear(500,6)

        )

        

        

    def train(self,train_loader,val_loader,epoch,waiting,optimizer):

        self.cuda()

        best_val_loss=1000

        crit = nn.CrossEntropyLoss()

        for i in range(epoch):

            train_loss = 0

            val_loss = 0

            for xx,yy in train_loader:

                xx = xx.cuda()

                yy=yy.cuda()

                optimizer.zero_grad()

                y_pred = self.forward(xx)

                loss = crit(y_pred,yy)

                train_loss += loss

                loss.backward()

                optimizer.step()

            train_loss = train_loss/len(train_loader)

            with torch.no_grad():

                for xx,yy in val_loader:

                    xx, yy = xx.cuda(), yy.cuda()

                    y_pred = self.forward(xx)

                    loss = crit(y_pred,yy)

                    val_loss += loss

                val_loss = val_loss/len(val_loader)

                

                if best_val_loss>val_loss:

                    torch.save(self.state_dict(), "../best_model.py")

                    best_val_loss = val_loss

                    wait=waiting

                else:

                    wait -=1

                    if wait==0:

                        break

            print("train loss:", float(train_loss), "___best val loss:",float(best_val_loss), "___remaining:", wait)

    

    def forward(self, x):

        return  self.estimator(self.properties(x).view(x.size(0), -1))
clf = Model()

optimizer = torch.optim.Adam(clf.parameters(), lr=0.001)

clf.train(train_loader,val_loader,10,4,optimizer)
from sklearn.metrics import classification_report

clf.load_state_dict(torch.load("../best_model.py"))

y_true = []

y_pred = []

for xx,yy in test_loader:

    out = clf.forward(xx.cuda())

    for i in out:

        y_pred.append(int(i.argmax()))

    for i in yy:

        y_true.append(int(i))

print(classification_report(y_pred,y_true))