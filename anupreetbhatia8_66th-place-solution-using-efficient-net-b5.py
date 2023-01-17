# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

df = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')

# df
test_df=pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/test.csv')

# test_df.head()
# df['image_id']=df['image_id']+".jpg"

# test_df['image_id']=test_df['image_id']+".jpg"
import torch

import torchvision.models as models

from PIL import Image

import torchvision.transforms as transforms

import os

from torchvision import datasets

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.utils.data import Dataset, DataLoader

import cv2                

from PIL import Image

from sklearn.metrics import accuracy_score

from albumentations import *

from albumentations.pytorch import ToTensor

!pip install --upgrade efficientnet-pytorch
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import train_test_split

# train_df,valid_df=train_test_split(df,test_size=0.2,shuffle=True,random_state=23,stratify=df.iloc[:,1:])
train_df=df
train_df.reset_index(drop=True,inplace=True)

# valid_df.reset_index(drop=True,inplace=True)

test_df.reset_index(drop=True,inplace=True)
class CustomDataset(Dataset):

    def __init__(self,df,root_dir,transform=None,iftest=False):

        self.df=df

        self.root_dir=root_dir

        self.transform=transform

        self.iftest=iftest

    

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self,idx):

        if torch.is_tensor(idx):

            idx=idx.tolist()

        img_name=self.root_dir+self.df.iloc[idx,0]+'.jpg'

#         print(img_name)

        image= cv2.imread(img_name,cv2.IMREAD_COLOR)

#         image= cv2.imread(img_name)

#         print(img_name,image)

        image= cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

#         image = Image.fromarray(image)

#         print(type(image))

        if self.transform:

            image=self.transform(image=image)['image']

        if self.iftest:

            return image

        labels=torch.tensor(np.argmax(self.df.iloc[idx,1:].values))

#         labels=np.asarray(labels)

#         labels=torch.from_numpy(labels.astype(np.int32))

#         labels=labels.unsqueeze(-1)

#         print(labels.shape)

#         sample={'image':image,'labels':labels}

        return (image,labels)

        

        
IMSIZE=545

IMSIZE=EfficientNet.get_image_size('efficientnet-b5')
print(IMSIZE)
train_dataset=CustomDataset(df=train_df,root_dir='/kaggle/input/plant-pathology-2020-fgvc7/images/',

                     transform=Compose([augmentations.transforms.Resize(height=IMSIZE,width=IMSIZE,always_apply=True),

                                                  HorizontalFlip(p=0.5),

                                                  VerticalFlip(p=0.5),

                                                  ShiftScaleRotate(rotate_limit=25.0,p=0.7),

                                                  OneOf([IAAEmboss(p=1),IAASharpen(p=1),Blur(p=1)],p=0.5),

                                                  IAAPiecewiseAffine(p=0.5),

                                                   Normalize((0.485,0.456,0.406),

                                                                      (0.229,0.224,0.225),always_apply=True),

                                                  ToTensor()

                                                  ]))
# valid_dataset=CustomDataset(df=valid_df,root_dir='/kaggle/input/plant-pathology-2020-fgvc7/images/',

#                      transform=Compose([augmentations.transforms.Resize(height=IMSIZE,width=IMSIZE,always_apply=True),

#                                                    Normalize((0.485,0.456,0.406),

#                                                                       (0.229,0.224,0.225),always_apply=True),

#                                                     ToTensor()

#                                                   ]))
test_dataset=CustomDataset(df=test_df,root_dir='/kaggle/input/plant-pathology-2020-fgvc7/images/',

                     transform=Compose([augmentations.transforms.Resize(height=IMSIZE,width=IMSIZE,always_apply=True),

                                                  Normalize((0.485,0.456,0.406),

                                                                      (0.229,0.224,0.225),always_apply=True),

                                                    ToTensor()

                                                  ]),iftest=True)
BATCH_SIZE=4
train_loader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)

# valid_loader=DataLoader(valid_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=2)

test_loader=DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=2)
use_cuda = torch.cuda.is_available()

if use_cuda:

    device='cuda:0'

use_tpu=False

use_device=True

if use_tpu:

    device='idk'
# model_transfer=models.densenet161(pretrained=True)

# # for param in model_transfer.parameters():

# #     param.requires_grad=False

# print(model_transfer)

# model_transfer.classifier=nn.Sequential(nn.Linear(model_transfer.classifier.in_features,1000),

#                                         nn.ReLU(),

#                                         nn.Dropout(p=0.5),

#                                         nn.Linear(1000,4))

# # nn.init.kaiming_normal_(model_transfer.classifier.weight, nonlinearity='relu')

# if use_device:

#     model_transfer = model_transfer.to(device)
# NEPOCHS=30

# print(IMSIZE)

# criterion_transfer = nn.CrossEntropyLoss()

# learning_rate=5e-4*np.logspace(0,1.5,9)

# learning_rate=learning_rate[2]

# learning_rate=8e-4

# optimizer_transfer = optim.AdamW(model_transfer.parameters(),learning_rate,weight_decay=1e-3)

# num_train_steps = int(len(train_dataset) / BATCH_SIZE * NEPOCHS)

# from transformers import get_cosine_schedule_with_warmup

# scheduler = get_cosine_schedule_with_warmup(optimizer_transfer, num_warmup_steps=len(train_dataset)/BATCH_SIZE*5, num_training_steps=num_train_steps)

# # optimizer_transfer = torch.optim.Adam(model_efficient.parameters())

# # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_transfer, 'max', patience = 3,verbose=True,min_lr=0.00001)
def train(n_epochs,train_loader,valid_loader,model,optimizer,criterion,use_device,save_path,final_train=False,ifsched=False):

    for epoch in range(1,n_epochs+1):

        train_loss=0.0

        valid_loss=0.0

        labels_for_acc=[]

        output_for_acc=[]

        labels_for_accv=[]

        output_for_accv=[]

        model.train()

        for batch_idx,(data,target) in enumerate(train_loader):

#             print(type(data),type(target))

            if use_device:

                data,target=data.to(device),target.to(device)

            optimizer.zero_grad()

            output=model(data)

            loss=criterion(output,target)

            train_loss+=loss.item()*data.size(0)

            loss.backward()

            optimizer.step()

            if ifsched:

                    scheduler.step()

            labels_for_acc=np.concatenate((labels_for_acc,target.cpu().numpy()),0)

            output_for_acc=np.concatenate((output_for_acc,np.argmax(output.cpu().detach().numpy(),1)),0)

        train_loss=train_loss/len(train_loader.dataset)

        train_acc=accuracy_score(labels_for_acc,output_for_acc)

        if not final_train:

            with torch.no_grad():

                model.eval()

                for batch_idx,(data,target) in enumerate(valid_loader):

                    if use_device:

                        data,target=data.to(device),target.to(device)

                    output=model(data)

                    loss=criterion(output,target)

                    valid_loss+=loss.item()*data.size(0)

                    labels_for_accv=np.concatenate((labels_for_accv,target.cpu().numpy()),0)

                    output_for_accv=np.concatenate((output_for_accv,np.argmax(output.cpu().detach().numpy(),1)),0)

                valid_loss=valid_loss/len(valid_loader.dataset)

                valid_acc=accuracy_score(labels_for_accv,output_for_accv)

                print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tTrain Acc: {:.6f} \tValidation Acc: {:.6f}'.format(

                epoch, 

                train_loss,

                valid_loss,

                train_acc,

                valid_acc

                ))

        if final_train:

            print('Epoch: {} \tTraining Loss: {:.6f} \tTrain Acc: {:.6f} '.format(

                epoch, 

                train_loss,

                train_acc

                ))

#     return model
# train(NEPOCHS, train_loader,valid_loader, model_transfer, optimizer_transfer, criterion_transfer, use_device, 'model_transfer.pt',ifsched=True,final_train=False)

from efficientnet_pytorch import EfficientNet

model_efficient=EfficientNet.from_pretrained('efficientnet-b7')
# for param in model_efficient.parameters():

#     param.requires_grad=False

# print(model_transfer)

model_efficient._fc=nn.Sequential(nn.Linear(model_efficient._fc.in_features,1000,bias=True),

                                 nn.ReLU(),

                                 nn.Dropout(p=0.5),

                                 nn.Linear(1000,4,bias=True))

# nn.init.kaiming_normal_(model_efficient._fc.weight, nonlinearity='relu')

if use_device:

    model_efficient = model_efficient.to(device)
NEPOCHS=40

print(IMSIZE)

criterion_transfer = nn.CrossEntropyLoss()

# learning_rate=5e-4*np.logspace(0,1.5,9)

# learning_rate=learning_rate[2]

learning_rate=8e-4

optimizer_transfer = optim.AdamW(model_efficient.parameters(),learning_rate,weight_decay=1e-3)

num_train_steps = int(len(train_dataset) / BATCH_SIZE * NEPOCHS)

from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(optimizer_transfer, num_warmup_steps=len(train_dataset)/BATCH_SIZE*5, num_training_steps=num_train_steps)

# optimizer_transfer = torch.optim.Adam(model_efficient.parameters())

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_transfer, 'max', patience = 3,verbose=True,min_lr=0.00001)

train(NEPOCHS, train_loader,None, model_efficient, optimizer_transfer, criterion_transfer, use_device, 'model_transfer.pt',ifsched=True,final_train=True)
def test(model,test_loader,use_device):

    preds_for_output=np.zeros((1,4))

    with torch.no_grad():

        model.eval()

        for images in test_loader:

            if use_device:

                images=images.to(device)

            preds=model(images)

            preds_for_output=np.concatenate((preds_for_output,preds.cpu().detach().numpy()),0)

    return preds_for_output

        

        
num_runs=5

import scipy

subs=[]

for i in range(num_runs):

    out=test(model_efficient,test_loader,use_device)

    output=pd.DataFrame(scipy.special.softmax(out,1),columns=['healthy','multiple_diseases','rust','scab'])

    output.drop(0,inplace=True)

    output.reset_index(drop=True,inplace=True)

    subs.append(output)



sub_eff=sum(subs)/num_runs
# test_df['image_id']=test_df['image_id'].str.replace('.jpg','')
sub1=sub_eff.copy()

sub1['image_id']=test_df.image_id

sub1=sub1[['image_id','healthy','multiple_diseases','rust','scab']]

sub1.to_csv('sub_densenet.csv',index=False)

sub1.head()