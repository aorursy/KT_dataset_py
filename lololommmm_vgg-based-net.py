# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim



from torchvision import datasets

from torchvision import transforms,utils

from PIL import Image

import torch.utils.data as Data



import torchvision

import numpy as np



import torch.nn as nn

from torch.autograd import Variable

import matplotlib.pyplot as plt

from torchvision import transforms

import random

class MyResize(object):

    def __init__(self, width, height):

        self.width = width

        self.height = height

    def __call__(self, img):

        raw_width, raw_height = img.size

        ratio = min(self.height/raw_height, self.width/raw_width)

        twidth, theight = (min(int(ratio * raw_width), self.width - 15), min(int(ratio * raw_height), self.height - 15))

        img = img.resize((twidth, theight), Image.ANTIALIAS)

        # 拼接图片，补足边框 居中

        ret = Image.new('L',(self.width, self.height), 255)

        ret.paste(img, (int((self.width-twidth)/2),int((self.height-theight)/2)))

        return ret

    

from torchvision import transforms



class picPull():

    def __init__(self, output_size, lower_bound, upper_bound):

        self.os = output_size

        self.lower_bound = lower_bound

        self.upper_bound = upper_bound

    def __call__(self, img):

        choice = random.uniform(self.lower_bound, self.upper_bound)

        img = MyResize(int(self.os[0]*choice), int(self.os[1]*choice)).__call__(img)

        if choice < 1:

            ret = Image.new('L', (self.os[0], self.os[1]), 255)

            ret.paste(img, (int((self.os[0]-self.os[0]*choice)/2),int((self.os[1]-self.os[1]*choice)/2)))

            return ret

        else:

            ret = transforms.CenterCrop(self.os).__call__(img)

            return ret



tf = transforms.Compose([

    MyResize(64, 64),

    picPull((64, 64), 0.5, 1.5),transforms.RandomRotation(180, fill=255),

    transforms.RandomHorizontalFlip(0.5),

    transforms.RandomVerticalFlip(0.5),

    transforms.RandomResizedCrop((64,64), scale=(0.6, 1.4), ratio=(0.75,1.25)),

    transforms.ToTensor()

])

tf2 = transforms.Compose([

    MyResize(64, 64),

    picPull((64, 64), 0.5, 1.5),transforms.RandomRotation(180, fill=255),

    transforms.RandomHorizontalFlip(0.5),

    transforms.RandomVerticalFlip(0.5),

    transforms.RandomResizedCrop((64,64), scale=(0.6, 1.4), ratio=(0.75,1.25)),

    transforms.ToTensor()

])

        

train_data = datasets.ImageFolder('../input/gnt123/tr_data', transform=tf)

test_data = datasets.ImageFolder('../input/gnt123/ts_data', transform=tf2)

batch_size = 64

train_loader = Data.DataLoader(train_data,batch_size=batch_size, num_workers=6, shuffle=True)

test_loader = Data.DataLoader(test_data,batch_size=batch_size, num_workers=6, shuffle=True)

def loadcheckpoint(model, optimizer, loc):

    data = torch.load(loc)

    model.load_state_dict(data['model'])

    optimizer.load_state_dict(data['optimizer'])

def set_checkpoint(checkpoint_name, model, optimizer):

    torch.save({

        'model': model.state_dict(),

        'optimizer': optimizer.state_dict()

    }, checkpoint_name);
class Net(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone=nn.Sequential(

            nn.Conv2d(1, 64, 3, 1, padding=1),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, 1, padding=1),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2, padding=1),

            

            nn.Conv2d(64, 128, 3, 1, padding=1),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, 1, padding=1),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2, padding=1),

            

            nn.Conv2d(128, 256, 3, 1, padding=1),

            nn.BatchNorm2d(256),

            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, 1, padding=1),

            nn.BatchNorm2d(256),

            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, 1, padding=1),

            nn.BatchNorm2d(256),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2, padding=1),

            

            nn.Conv2d(256, 512, 3, 1, padding=1),

            nn.BatchNorm2d(512),

            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, 3, 1, padding=1),

            nn.BatchNorm2d(512),

            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, 3, 1, padding=1),

            nn.BatchNorm2d(512),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2, padding=1),

        )

        self.linear = nn.Linear(12800, 3755, bias=True)

 

    def forward(self, x):

        x = self.backbone(x)

        x = torch.flatten(x, 1)

        x = self.linear(x)

        x = F.log_softmax(x, dim=1)

        return x


def train(model,device,train_loader,optimizer,epoch):

    model.train()

    total_loss = 0.

    correct = 0.

    for idx,(data,target) in enumerate(train_loader):

        data,target = data.to(device),target.to(device)

        

        output = model(data)

        loss = F.nll_loss(output,target)

        total_loss+= F.nll_loss(output,target,reduction="sum").item()

        pred = model(data).argmax(dim = 1)

        correct += pred.eq(target.view_as(pred)).sum().item()

        

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        

    total_loss /= len(train_loader.dataset)

    acc = correct /len(train_loader.dataset)*100

    print("Train Epoch:{},Loss:{},Accuracy:{}".format(epoch,loss.item(),acc))
def test(model,device,test_loader):

    model.eval()

    total_loss = 0.

    correct = 0.

    with torch.no_grad():

        for idx,(data,target) in enumerate(test_loader):

            data,target = data.to(device),target.to(device)

        

            output = model(data)

            total_loss+= F.nll_loss(output,target,reduction="sum").item()

            pred = model(data).argmax(dim = 1)

            correct += pred.eq(target.view_as(pred)).sum().item()

    

               

    total_loss /= len(test_loader.dataset)

    acc = correct /len(test_loader.dataset)*100.

    print("Test loss:{},Accuracy:{}".format(total_loss,acc))    
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#device = torch.device("cpu")

lr = 0.001

momentum = 0.5

model = Net().to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=lr)

loadcheckpoint(model, optimizer, '../input/12step/12')

num_epochs = 13



for epoch in range(num_epochs):

    model.train()

    total_loss = 0.

    correct = 0.

    for data,target in tqdm(train_loader):

        data,target = data.to(device),target.to(device)

        

        output = model(data)

        loss = F.nll_loss(output,target)

        total_loss+= F.nll_loss(output,target,reduction="sum").item()

        pred = model(data).argmax(dim = 1)

        correct += pred.eq(target.view_as(pred)).sum().item()

        

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        

    total_loss /= len(train_loader.dataset)

    acc = correct /len(train_loader.dataset)*100.

    print("Train Epoch:{},Loss:{},Accuracy:{}".format(epoch,loss.item(),acc))

    

    test(model,device,test_loader)

    if (epoch % 6) == 0:

        set_checkpoint('./'+str(epoch), model, optimizer)


