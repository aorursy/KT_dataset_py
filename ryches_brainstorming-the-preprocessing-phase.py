# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt

import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))

from skimage import io

# Any results you write to the current directory are saved as output.
# importing pytorch modules

import torch

import torch.nn.functional as F

import torch.nn as nn

from torch import optim

import torchvision

import torchvision.datasets as datasets

import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Dataset

from torchvision import models

from time import time
tic=time()

print("Time module testing")

tac=time()

print("duration: ", tac-tic)
torch.tensor([0], dtype = torch.long)
class DatasetCUSTOM(Dataset):

    

    def __init__(self, image_path, label_path, transform=None):

        self.image_path= image_path

        self.labels= pd.read_csv(label_path)

        self.transform = transform

        

    def __len__(self):

        return len(self.labels)

    

    def __getitem__(self, idx):

            

        img_name= os.path.join(self.image_path, self.labels.iloc[idx,0])

        image= io.imread(img_name+".png")

        image=np.array(image).astype(np.uint8)

        label = self.labels.iloc[idx,1] 

        if self.transform is not None:

            image = self.transform(image)

            

        return image, label
transform_train=transforms.Compose([

    transforms.ToPILImage(),

    transforms.Resize((229,229)),

    transforms.ToTensor(),

    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224, 0.225)),

    ])
images = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")["id_code"]
import cv2
os.mkdir("train_images_229")
for image in images:

    img = cv2.imread("../input/aptos2019-blindness-detection/train_images/" + image + ".png")

    img = cv2.resize(img, (229,229))

    cv2.imwrite("./train_images_229/" + image + ".png", img)
batch_size = 64

trainset=DatasetCUSTOM("./train_images_229/",

                       "../input/aptos2019-blindness-detection/train.csv",

                       transform=transform_train

                       )

trainloader=DataLoader(trainset,batch_size=batch_size,shuffle=True)

dataiter=iter(trainloader)

tic=time()

images, labels=dataiter.next()

tac=time()

print("Dataiter time:", tac-tic)
print(len(trainset))
labels
print(images.shape)

npimg=np.array(images[1])

img=np.transpose(npimg,(2,1,0))

print(img.shape)

plt.imshow(img)
class LeNet(nn.Module):

    def __init__(self):

        super(LeNet, self).__init__()

        self.cnn_model= nn.Sequential(

            nn.Conv2d(3,6,5), #(28,28,1)------>(24,24,6)

            nn.ReLU(),

            nn.MaxPool2d(2, stride=2), # (24,24,6)------>(12,12,6)

            nn.Conv2d(6,16,5), # (12,12,6)-------->(8,8,16)

            nn.ReLU(),

            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(16,32,3),

            nn.ReLU(),

            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(32,32,3),

            nn.ReLU(),

            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(32,32,3),

            nn.ReLU(),

            nn.MaxPool2d(2, stride=2)

            )

            

        self.fc_model=nn.Sequential(

            nn.Linear(800,120),

            nn.ReLU(),

            nn.Linear(120,84),

            nn.ReLU(),

            nn.Linear(84,5),

            nn.Softmax()

            )

            

        

    def forward(self,x):

#         print(x.shape)

        x = self.cnn_model(x)

        x = nn.Flatten()(x)

        x = self.fc_model(x)

#         print(x.shape)

        return x
def evaluation(dataloader, model):

    correct, total=0,0

    for data in dataloader:

        inputs, labels= data

        inputs, labels= inputs.to(device), labels.to(device)

        output= model(inputs)

        _, pred = torch.max(output,1)

        total+= labels.size()

        correct+= (pred==labels).sum().item()

    return 100*correct/total

        
device="cuda"
model=LeNet().to(device)

loss_fn=nn.CrossEntropyLoss()

opt=optim.Adam(model.parameters(), lr=0.0001)
for param in model.parameters():

    if param.requires_grad:

        print(param.shape)
loss_per_epoch=[]

loss_arr=[]

max_epochs=10

n_iters= np.ceil(3662/batch_size)

for epoch in range(max_epochs):

    for i, data in enumerate(trainloader,0):

        inputs, labels= data

        inputs, labels= inputs.cuda(), labels.cuda()

#         print(inputs.shape)

        opt.zero_grad()

        outputs = model(inputs)

        loss=loss_fn(outputs, labels)

        loss_arr.append(loss)

        loss.backward()

        opt.step()

        del inputs, labels, outputs

        torch.cuda.empty_cache()

        if i% 500==0:

            print("Epoch:%d, Iteration: %d/%d, Loss: %0.2f" %(epoch,i,n_iters,loss.item()))

    loss_per_epoch.append(loss.item())

plt.plot(loss_arr,'r')

plt.show()
images = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")["id_code"]
import zipfile

zf = zipfile.ZipFile('train_images_229.zip', mode='w')

for image in images:

    zf.write("./train_images_229/" + image + ".png")

    os.remove("./train_images_229/" + image + ".png")

zf.close()