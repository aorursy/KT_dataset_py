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
#Import package

import sys

import csv

import time

import numpy as np

import torch

import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader



#Read in train.csv and split data into training/validation set

def readfile(path):

    print("Reading File...")

    x_train = []

    x_label = []

    val_data = []

    val_label = []



    raw_train = np.genfromtxt(path, delimiter=',', dtype=str, skip_header=1)

    for i in range(len(raw_train)):

        tmp = np.array(raw_train[i, 1].split(' ')).reshape(1, 48, 48)

        if (i % 10 == 0):

            val_data.append(tmp)

            val_label.append(raw_train[i][0])

        else:

            x_train.append(tmp)

            x_train.append(np.flip(tmp, axis=2))    # simple example of data augmentation

            #x_train.append(np.flip(tmp, axis=1))

            #x_train.append(np.rot90(tmp,1,(1,2)))

            #x_label.append(raw_train[i][0])

            #x_label.append(raw_train[i][0])

            x_label.append(raw_train[i][0])

            x_label.append(raw_train[i][0])



    x_train = np.array(x_train, dtype=float) / 255.0

    val_data = np.array(val_data, dtype=float) / 255.0

    x_label = np.array(x_label, dtype=int)

    val_label = np.array(val_label, dtype=int)

    x_train = torch.FloatTensor(x_train)

    val_data = torch.FloatTensor(val_data)

    x_label = torch.LongTensor(x_label)

    val_label = torch.LongTensor(val_label)



    return x_train, x_label, val_data, val_label



x_train, x_label, val_data, val_label = readfile("../input/train.csv")    # 'train.csv'



#Wrapped as dataloader

train_set = TensorDataset(x_train, x_label)

val_set = TensorDataset(val_data, val_label)



batch_size = 256

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)

val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)



print('1')
#Model Construction

def gaussian_weights_init(m):

    classname = m.__class__.__name__

    if classname.find('Conv') != -1 and classname.find('Conv') == 0:

        m.weight.data.normal_(0.0, 0.02)





class Classifier(nn.Module):

    def __init__(self):

        super(Classifier, self).__init__()

        self.cnn = nn.Sequential(

            nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),  # [64, 48, 48]

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0),

            nn.Dropout(p=0.3), # [64, 24, 24]



            nn.Conv2d(64, 128, 3, 1, 1),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.MaxPool2d(2, 2, 0),      

            nn.Dropout(p=0.3),   # [128, 12, 12]

            

            nn.Conv2d(128, 256, 3, 1, 1),

            nn.BatchNorm2d(256),

            nn.ReLU(),

            nn.MaxPool2d(2, 2, 0),      

            nn.Dropout(p=0.3),   # [256, 6, 6]

            

            nn.Conv2d(256, 512, 3, 1, 1),

            nn.BatchNorm2d(512),

            nn.ReLU(),

            nn.MaxPool2d(2, 2, 0),      

            nn.Dropout(p=0.4),   # [512, 3, 3]

        )



        self.fc = nn.Sequential(  #fully connected layers

            nn.Linear(512*3*3, 512),

            nn.ReLU(),

            nn.Dropout(p=0.5),

            nn.Linear(512, 512),

            nn.ReLU(),

            nn.Dropout(p=0.5),

            nn.Linear(512, 7),

        )



        self.cnn.apply(gaussian_weights_init)

        self.fc.apply(gaussian_weights_init)



    def forward(self, x):

        out = self.cnn(x)

        out = out.view(out.size()[0], -1)

        return self.fc(out)



#Training

model = Classifier().cuda()

# print(model)

loss = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_acc = 0.0

num_epoch = 150



for epoch in range(num_epoch):

    epoch_start_time = time.time()

    train_acc = 0.0

    train_loss = 0.0

    val_acc = 0.0

    val_loss = 0.0



    model.train()

    for i, data in enumerate(train_loader):

        optimizer.zero_grad()



        train_pred = model(data[0].cuda())

        batch_loss = loss(train_pred, data[1].cuda())

        batch_loss.backward()

        optimizer.step()



        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())

        train_loss += batch_loss.item()



        progress = ('#' * int(float(i)/len(train_loader)*40)).ljust(40)

        print ('[%03d/%03d] %2.2f sec(s) | %s |' % (epoch+1, num_epoch, \

                (time.time() - epoch_start_time), progress), end='\r', flush=True)

    

    model.eval()

    for i, data in enumerate(val_loader):

        val_pred = model(data[0].cuda())

        batch_loss = loss(val_pred, data[1].cuda())



        val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())

        val_loss += batch_loss.item()



        progress = ('#' * int(float(i)/len(val_loader)*40)).ljust(40)

        print ('[%03d/%03d] %2.2f sec(s) | %s |' % (epoch+1, num_epoch, \

                (time.time() - epoch_start_time), progress), end='\r', flush=True)



    val_acc = val_acc/val_set.__len__()

    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \

            (epoch + 1, num_epoch, time.time()-epoch_start_time, \

             train_acc/train_set.__len__(), train_loss, val_acc, val_loss))

    

   



   
def readtestfile(path):

    print("Reading File...")

    x_test = []

    x_label = []



    raw_test = np.genfromtxt(path, delimiter=',', dtype=str, skip_header=1)

    for i in range(len(raw_test)):

        tmp = np.array(raw_test[i, 1].split(' ')).reshape(1, 48, 48)

        x_test.append(tmp)

        x_label.append(raw_test[i][0])



    x_test = np.array(x_test, dtype=float) / 255.0

    x_label = np.array(x_label, dtype=int)

    x_test = torch.FloatTensor(x_test)

    x_label = torch.LongTensor(x_label)



    return x_test, x_label





x_test, x_label = readtestfile("../input/test.csv")    # 'test.csv'

test_set = TensorDataset(x_test, x_label)

test_loader = DataLoader(test_set, num_workers=8)

print('1')



submission = [['id', 'label']]

for i, data in enumerate(test_loader):

        

        test_pred = model(data[0].cuda())

        pred = np.argmax(test_pred.cpu().data.numpy(), axis=1)[0]

        submission.append([i, pred])

with open('submission.csv', 'w') as submissionFile:

    writer = csv.writer(submissionFile)

    writer.writerows(submission)

    

print('Writing Complete!')