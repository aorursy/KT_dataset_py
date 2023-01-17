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

from PIL import Image

# Any results you write to the current directory are saved as output.
import cv2
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
class DatasetCUSTOM(Dataset):

    

    def __init__(self, image_path, label_path, transform=None):

        self.image_path= image_path

        self.labels= pd.read_csv(label_path)

        self.transform = transform

        

    def __len__(self):

        return len(self.labels)

    

    def __getitem__(self, idx):

#         if torch.is_tensor(idx):

#             idx=idx.tolist()

            

        img_name= os.path.join(self.image_path, self.labels.iloc[idx,0])

        image= Image.open(img_name+".png")

        image=np.array(image).astype(np.uint8)

#         print(image.shape)

#         image=np.transpose(image,(2,0,1)).astype(np.uint8)

#         image=torch.Tensor(image)

#         print(image.dtype, image.shape)

        label= self.labels.iloc[idx,1].toTensor()

        if self.transform is not None:

            image = self.transform(image)

            

        return image, label
transform_train=transforms.Compose([

#     transforms.ToPILImage(),

#     transforms.Resize((28,28)),

    transforms.ToTensor(),

    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224, 0.225)),

    ])
images_name = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")["id_code"]

os.mkdir("../input/train_images_229")
for image in images_name:

    img = cv2.imread("../input/aptos2019-blindness-detection/train_images/" + image + ".png")

    img = cv2.resize(img, (229,229))

    cv2.imwrite("../input/train_images_229/" + image + ".png", img)
os.listdir("../input/train_images_229/")
batch_size = 128

trainset=DatasetCUSTOM("../input/train_images_229/",

                       "../input/aptos2019-blindness-detection/train.csv",

                       transform=transform_train

                       )

trainloader=DataLoader(trainset,batch_size=batch_size,shuffle=False)

dataiter=iter(trainloader)

tic=time()

images, labels=dataiter.next()

tac=time()

print("Dataiter time:", tac-tic)
images, labels=iter(trainloader).next()
print(labels.size()[0])

npimg=np.array(images[1])

img=np.transpose(npimg,(2,1,0))

plt.imshow(img)
model = models.resnet50(pretrained=True)
for param in model.parameters():

    param.requires_grad=False
num_classes=5

final_in_features=model.fc.in_features

model.fc=nn.Linear(final_in_features,num_classes)

print(model.fc)
for param in model.parameters():

    if param.requires_grad:

        print(param.shape)
def evaluation(dataloader, model):

    correct, total=0,0

    for data in dataloader:

        inputs, labels= data

        inputs, labels= inputs.to(device), labels.to(device)

        output= model(inputs)

        _, pred = torch.max(output,1)

        total+= labels.size()[0]

        correct+= (pred==labels).sum().item()

    return 100*correct/total

        
device="cuda"
model=model.to(device)

loss_fn=nn.CrossEntropyLoss()
opt=optim.Adam(model.parameters(), lr=0.001)
batch_size= 128

trainloader=DataLoader(trainset,batch_size=batch_size,shuffle=False)
model.train()

loss_per_epoch=[]

loss_arr=[]

max_epochs=16

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

        if i% 2==0:

            print("Epoch:%d, Iteration: %d/%d, Loss: %0.2f" %(epoch,i,n_iters,loss.item()))

    loss_per_epoch.append(loss.item())

plt.plot(loss_per_epoch,'r')

plt.show()
model.eval()

print("Training Accuracy:", evaluation(trainloader,model))