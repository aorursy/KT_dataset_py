import torch

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

from torch.utils.data import DataLoader

import torchvision.datasets as datasets

import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader

import numpy as np

import pandas as pd

import os

import numpy

import glob

import cv2
#https://stackoverflow.com/questions/49537604/how-to-read-multiple-images-from-multiple-folders-in-python

folders = glob.glob('../input/flowers-recognition/flowers/flowers/*')

imagenames_list = []

for folder in folders:

    for f in glob.glob(folder+'/*.jpg'):

        imagenames_list.append(f)
def label_img(image):

    word_label = image.split('/')[4]

    if word_label == 'daisy':

        return [1,0,0,0,0]

    elif word_label == 'dandelion':

        return [0,1,0,0,0]

    elif word_label == 'rose':

        return [0,0,1,0,0]

    elif word_label == 'tulip':

        return [0,0,0,1,0]

    else:

        return [0,0,0,0,1]
train = []        



for image in imagenames_list:

    label = label_img(image)

    train.append([np.array(cv2.resize(cv2.imread(image),(224,224))), np.array(label)])

    np.random.shuffle(train)
X_numpy = np.array([i[0] for i in train])

Y_numpy = np.array([i[1] for i in train])
#select first couple of samples just for easy training

X_numpy = X_numpy[:64]

Y_numpy = Y_numpy[:64]
X_torch = torch.from_numpy(X_numpy)

Y_torch = torch.from_numpy(Y_numpy)



X_torch_flt = X_torch.float()
class VGGDataset(Dataset):

    

    def __init__(self):

        #data loading

        self.X = X_torch_flt

        self.Y = Y_torch

        self.n_samples = X_numpy.shape[0]

        

    def __getitem__(self, index):

        return self.X[index], self.Y[index]

    

    def __len__(self):

        return self.n_samples
dataset = VGGDataset()

dataloader = DataLoader(dataset = dataset, shuffle = True, batch_size = 32)
VGG16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
class VGG(nn.Module):

    def __init__(self, in_channels=3, num_classes=5):

        super(VGG,self).__init__()

        self.in_channels = in_channels

        self.conv_layers = self.create_conv_layers(VGG16)

        self.fcs = nn.Sequential(

            nn.Linear(512*7*7, 4096),

            nn.ReLU(),

            nn.Dropout(p=0.5),

            nn.Linear(4096,4096),

            nn.ReLU(),

            nn.Dropout(p=0.5),

            nn.Linear(4096, num_classes)

        )

        

    def forward(self, x):

        x = self.conv_layers(x)

        x = x.reshape(x.shape[0], -1)

        x = self.fcs(x)

        return F.log_softmax(x, dim=1)

    

    def create_conv_layers(self, architecture):

        layers = []

        in_channels = self.in_channels

        

        for x in architecture:

            if type(x)  == int:

                out_channels = x

                

                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,

                                    kernel_size=(3,3), stride=(1,1), padding=(1,1)),

                          nn.BatchNorm2d(x),

                          nn.ReLU()]

                in_channels = x

            elif x == 'M':

                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]

                    

        return nn.Sequential(*layers)
model = VGG()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

criterion = nn.CrossEntropyLoss()
for epoch in range(1):

    for i, (X, Y) in enumerate(dataloader):

    

        #Forward Pass: Compute predicted Y by passing X through the model

        #reshaping from (224,224,3) to (3,224,224): https://stackoverflow.com/questions/56789038/runtimeerror-given-groups-1-weight-of-size-64-3-3-3-expected-input4-50

        y_pred = model(X.permute(0, 3, 1, 2))

    

        #Compute and print loss

        #multi-class output not support: https://discuss.pytorch.org/t/runtimeerror-multi-target-not-supported-newbie/10216/4

        loss = criterion(y_pred, torch.max(Y, 1)[1])

        print(epoch, loss.item())

    

        #Zero Gradients, Backwards Pass, and Update Weights

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()