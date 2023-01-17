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

import torch.nn as nn

import torch.nn.functional as F

import torchvision

from torchvision import transforms

from torchvision.datasets import ImageFolder

from torch.utils.data import DataLoader

from PIL import Image

from sklearn.metrics import classification_report

from IPython.display import display

# Any results you write to the current directory are saved as output.
seed = 255

np.random.seed(seed)

torch.manual_seed(seed)
train_dir = "../input/fruits-360_dataset/fruits-360/Training/"

test_dir = "../input/fruits-360_dataset/fruits-360/Test/"
class ConvNet(nn.Module):

    def __init__(self):

        super().__init__()

        # 100*100 * 3

        self.l1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=7)

        # 94*94 * 20

        self.l2 = nn.MaxPool2d(kernel_size=2)

        # 47*47 * 20

        self.l3 = nn.Conv2d(in_channels=20, out_channels=100, kernel_size=8)

        # 40*40 * 100

        self.l4 = nn.MaxPool2d(kernel_size=4)

        # 10*10 * 100(10_000)

        self.l5 = nn.Linear(in_features=10*10*100, out_features=103)

        

    

    def forward(self, x):

        x = self.l1(x)

        x = F.relu(self.l2(x))

        x = self.l3(x)

        x = F.relu(self.l4(x))

        x = x.view(-1, 10*10*100)

        x = self.l5(x)

        return x

    

cnn = ConvNet().cuda()

cnn
train_images = []

count = 0

for i in os.listdir(train_dir):

    for j in os.listdir(train_dir + i + "/"):

        train_images.append((i + "/" + j, count))

    count += 1
train_list = os.listdir(train_dir + "Kiwi/")[:5]

_mean = [.485, .456, .406]

_std = [.229, .224, .225]

trans = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(_mean, _std)

])

train_ds = ImageDataset(train_dir, trans)

train_dl = DataLoader(train_ds, batch_size=100, shuffle=True)
optim = torch.optim.Adam(cnn.parameters(), lr=0.0001)

criterion = nn.CrossEntropyLoss()
cnn.train()

for epoche in range(1):

    counter = 0.

    for xx, yy in train_dl:

        xx, yy = xx.cuda(), yy.cuda()

        optim.zero_grad()

        out = cnn(xx)

        loss = criterion(out, yy)

        loss.backward()

        optim.step()

        counter += 100

        procent = counter/len(train_dl)

        if(procent % 5 == 0):

            print(procent)
test_ds = ImageDataset(test_dir, trans)

test_dl = DataLoader(test_ds, batch_size =100, shuffle=False)
cnn.eval()

with torch.no_grad():

    all_pred = []

    all_y = []

    counter = 0.

    for xx, yy in test_dl:

        xx = xx.cuda()

        pred = cnn(xx).argmax(1).tolist()

        all_pred.extend(pred)

        all_y.extend(yy.tolist())

        counter += 100

        procent = counter/len(test_dl)

        print(procent)

        

    print(classification_report(all_y, all_pred))