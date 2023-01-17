import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import torch

from torchvision import transforms,datasets

from PIL import Image

import random

import torch.utils.data as Data

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from PIL import Image

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook as tqdm
class MyResize(object):

    def __init__(self, width, height):

        self.width = width

        self.height = height

    def __call__(self, img):

        raw_width, raw_height = img.size

        ratio = min(self.height/raw_height, self.width/raw_width)

        twidth, theight = (min(int(ratio * raw_width), self.width), min(int(ratio * raw_height), self.height))

        img = img.resize((twidth, theight), Image.ANTIALIAS)

        # 拼接图片，补足边框 居中

        ret = Image.new('L',(self.width, self.height), 255)

        ret.paste(img, (int((self.width-twidth)/2),int((self.height-theight)/2)))

        return ret

    

class EnainTranform(object):

    def __init__(self):

        pass

    def __call__(self, img):

        return img

    

class MyTransform(object):

    def __init__(self):

        self.o1 = transforms.Compose([

            MyResize(64, 64),

            

            transforms.RandomChoice([

                #EnainTranform(),

                transforms.RandomRotation(25,fill=255),

                transforms.RandomVerticalFlip(),

                transforms.RandomHorizontalFlip()

            ]),

            transforms.ToTensor()

        ])

        self.o2 = transforms.Compose([

            MyResize(64, 64),

            transforms.ToTensor()

        ])

    def __call__(self, img):

        seed = random.uniform(0,1);

        #print(seed)

        if seed <= 0.75:

            return self.o1(img)

        else:

            width = random.uniform(0.5, 2.5) * img.size[0]

            height = random.uniform(0.5, 2.5) * img.size[1]

            width = int(width)

            height = int(height)

            img = img.resize((width, height), Image.ANTIALIAS)

            

            return self.o2(img)

        

test_data = datasets.ImageFolder('../input/gnt123/ts_data', transform = MyTransform())

train_data = datasets.ImageFolder('../input/gnt123/tr_data', transform = MyTransform())

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

lr = 0.0001

momentum = 0.5

model = Net().to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=lr)

loadcheckpoint(model, optimizer, '../input/dadada/3')

               

num_epochs = 4



for epoch in range(num_epochs):

    model.train()

    total_loss = 0.

    correct = 0.

    for data,target in train_loader:

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

    if (epoch % 3) == 0:

        set_checkpoint('./'+str(epoch), model, optimizer)



'''

import matplotlib.pyplot as plt

for a,b in train_loader:

    print(a.shape)

    plt.imshow(a[0][0])

    break;

'''