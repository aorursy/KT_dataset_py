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

from torchvision.datasets import ImageFolder

import torch

import torch.nn

import torch.nn.functional

from PIL import Image

from torch.utils.data import DataLoader

from torchvision import transforms

from sklearn.metrics import classification_report

print(len(os.listdir("../input/fruits-360_dataset/fruits-360/Training")))
mean = [.485, .456, .406]

std = [.229, .224, .225]

transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(mean, std)

])



def load_dataset(data_path):

    train_dataset = ImageFolder(

        root=data_path,

        transform=transform

    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=64,

        shuffle=True

    )

    return train_loader

train_loader = load_dataset("../input/fruits-360_dataset/fruits-360/Training/")

test_loader = load_dataset("../input/fruits-360_dataset/fruits-360/Test/")
class Model(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.l1 = torch.nn.Conv2d(3, 20, 5)#100

        self.l2 = torch.nn.MaxPool2d(3)#94

        self.l3 = torch.nn.Conv2d(20, 100, 3)#32

        self.l4 = torch.nn.MaxPool2d(3)#30

        self.l5 = torch.nn.Linear(10000, 500)#10

        self.l6 = torch.nn.Linear(500, 103)

        

    def forward(self, x):

        out = self.l1(x)

        out = torch.nn.functional.relu(self.l2(out))

        out = self.l3(out)

        out = torch.nn.functional.relu(self.l4(out)).view(-1,10000)

        out = torch.nn.functional.relu(self.l5(out))

        return self.l6(out)
model = Model().cuda()
model.train()

optim = torch.optim.Adam(model.parameters(), lr=0.001)

criterion = torch.nn.CrossEntropyLoss()

for i in range(2):

    num = 0

    for x, y in train_loader:

        optim.zero_grad()

        x, y = x.cuda(), y.cuda()

        out = model(x)

        loss = criterion(out, y)

        loss.backward()

        optim.step()

        if num%100 == 0:

            print(i , " - " , loss.item())
model.eval()

outs = []

ys = []

with torch.no_grad():

    for x, y in test_loader:

        x, y = x.cuda(), y.cuda()

        ys.extend(y.tolist())

        out = torch.nn.functional.softmax(model(x), dim=1).argmax(dim=1)

        outs.extend(out.tolist())

        

print(classification_report(ys, outs))