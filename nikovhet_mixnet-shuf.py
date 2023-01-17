!pip install -U git+https://github.com/albu/albumentations;

!pip install -U git+https://github.com/rwightman/pytorch-image-models

from google.colab import drive

drive.mount('/content/drive')
import numpy as np

import pandas as pd

import torch

import random

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

import os

import shutil

import cv2

import time

import torchvision

import PIL

from PIL import Image

from tqdm import tqdm_notebook as tqdm

from torch.utils.data import Dataset, DataLoader, random_split

import albumentations 

from albumentations.pytorch import ToTensorV2 as AT

import matplotlib.pyplot as plt

import timm

import zipfile
#!unzip 'drive/My Drive/ColabNotebooks/Kolab/ML/dataset/grow' -d 'drive/My Drive/ColabNotebooks/Kolab/ML/dataset/train/train'
with zipfile.ZipFile('drive/My Drive/ColabNotebooks/Kolab/ML/dataset/korpus.zip','r')as zipf:

  zipf.extractall('drive/My Drive/ColabNotebooks/Kolab/ML/dataset/data/')

PATH = 'drive/My Drive/ColabNotebooks/Kolab/ML/dataset/data/'

train_path = os.path.join(PATH, "train/train/")

test_path = os.path.join(PATH, "test/test/")

# путь к данным

sample_submission = pd.read_csv("drive/My Drive/ColabNotebooks/Kolab/ML/dataset/sample_submission.csv")

#------------------------------------------------------------------------------

batch_size = 32

num_workers = os.cpu_count()

#img_size = 256

img_size = 224

#------------------------------------------------------------------------------

train_list = os.listdir(train_path)

test_list = os.listdir(test_path)

#print(len(train_list), len(test_list))

class ChartsDataset(Dataset):

    

    def __init__(self, path, img_list, transform=None, mode='train'):

        self.path = path

        self.img_list = img_list

        self.transform = transform

        self.mode = mode

    

    def __len__(self):

        return len(self.img_list)

    

    def __getitem__(self, idx):

        image_name = self.img_list[idx]

        

        if image_name.split(".")[-1] == "gif":

           gif = cv2.VideoCapture(self.path + image_name)

           _, image = gif.read()

        else:

            pil_image = PIL.Image.open(self.path + image_name).convert('RGB') 

            image = np.array(pil_image) 

            

        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        

        label = 0 #just_image

        if "bar_chart" in image_name:

            label = 1

        elif "diagram" in image_name:

            label = 2

        elif "flow_chart" in image_name:

            label = 3

        elif "graph" in image_name:

            label = 4

        elif "growth_chart" in image_name:

            label = 5

        elif "pie_chart" in image_name:

            label = 6

        elif "table" in image_name:

            label = 7

            

        if self.transform:

            augmented = self.transform(image=image)

            image = augmented["image"]

        

        if self.mode == "train":

            return image, label

        else:

            return image, image_name



data_transforms = albumentations.Compose([

    albumentations.Resize(img_size, img_size),

    albumentations.Normalize(),

    AT()

    ])





data_transforms_test = albumentations.Compose([

    albumentations.Resize(img_size, img_size),

    albumentations.Normalize(),

    AT()

    ])

trainset = ChartsDataset(train_path, train_list,  transform=data_transforms)

testset = ChartsDataset(test_path, test_list,  transform=data_transforms_test, mode="test")

valid_size = int(len(train_list) * 0.1)

train_set, valid_set = torch.utils.data.random_split(trainset, 

                                    (len(train_list)-valid_size, valid_size))



#создаем даталоадеры для всех 3х подвыборок.

trainloader = torch.utils.data.DataLoader(train_set, 

                                        batch_size=batch_size, shuffle=True,

                                        pin_memory=False,

                                        num_workers = num_workers)



validloader = torch.utils.data.DataLoader(valid_set, 

                                        batch_size=batch_size, shuffle=True,

                                        pin_memory=False,

                                        num_workers = num_workers)



testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size,

                                         num_workers = num_workers)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





model= timm.create_model('mixnet_xl', pretrained=False)



for param in model.parameters():

    param.requires_grad = False



in_features = model.classifier.in_features

#model.global_pool(output_size=1,)

model.classifier = nn.Linear(in_features, 8).to(device)

#model.eval()

model.load_state_dict(torch.load('drive/My Drive/ColabNotebooks/Kolab/ML/workmodel/mbest2.pt'))
model = torchvision.models.densenet161(pretrained=True)

for param in model.parameters():

    param.requires_grad = False

model

in_features = model.classifier.in_features

model.classifier = nn.Linear(in_features, 8)
model.load_state_dict(torch.load('drive/My Drive/ColabNotebooks/Kolab/ML/workmodel/mbest.pt'))

#model.eval()
def train_model(model_conv, train_loader, valid_loader, criterion, optimizer, sheduler, n_epochs):

    model_conv.to(device)

    valid_loss_min = np.Inf

    patience = 5

    # сколько эпох ждем до отключения

    p = 0

    # иначе останавливаем обучение

    stop = False



    # количество эпох

    for epoch in range(1, n_epochs+1):

        print(time.ctime(), 'Epoch:', epoch)



        train_loss = []



        for batch_i, (data, target) in enumerate(tqdm(train_loader)):

            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            output = model_conv(data)

            loss = criterion(output, target)

            train_loss.append(loss.item())

            loss.backward()

            optimizer.step()

        # запускаем валидацию

        model_conv.eval()

        val_loss = []

        for batch_i, (data, target) in enumerate(valid_loader):

            data, target = data.to(device), target.to(device)

            output = model_conv(data)

            loss = criterion(output, target)

            val_loss.append(loss.item()) 



        print(f'Epoch {epoch}, train loss: {np.mean(train_loss):.4f}, valid loss: {np.mean(val_loss):.4f}.')



        valid_loss = np.mean(val_loss)

        scheduler.step(valid_loss)

        if valid_loss <= valid_loss_min:

            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

            valid_loss_min,

            valid_loss))

            torch.save(model_conv.state_dict(), 'model.pt')

            torch.save(model_conv.state_dict(), 'drive/My Drive/ColabNotebooks/Kolab/ML/model.pt')

            valid_loss_min = valid_loss

            p = 0



        # проверяем как дела на валидации

        if valid_loss > valid_loss_min:

            p += 1

            print(f'{p} epochs of increasing val loss')

            if p > patience:

                print('Stopping training')

                stop = True

                break        



        if stop:

            break

    return model_conv, train_loss, val_loss
#----------------------------------------------------------------------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,weight_decay=1e-6)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=2,)

#----------------------------------------------------------------------------------------------
torch.cuda.empty_cache()

model_resnet, train_loss, val_loss = train_model(model, trainloader, validloader, criterion,optimizer, scheduler, 

                                                 n_epochs=20,)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model1= timm.create_model('mixnet_xl', pretrained=False).to(device)

#model3 = torchvision.models.densenet161(pretrained=False).to(device)

model2 = torchvision.models.shufflenet_v2_x1_0(pretrained=False).to(device)



model1.eval()

model3.eval()

#model3.eval()

in_features = model1.classifier.in_features

model1.classifier = nn.Linear(in_features, 8).to(device)

model1.load_state_dict(torch.load('drive/My Drive/ColabNotebooks/Kolab/ML/workmodel/mbest.pt'))



#in_features = model3.classifier.in_features

#model3.classifier = nn.Linear(in_features, 8).to(device)

#model3.load_state_dict(torch.load('drive/My Drive/ColabNotebooks/Kolab/ML/workmodel/dv2.pt'))



in_features = model2.fc.in_features

model2.fc = nn.Linear(in_features, 8).to(device)

model2.load_state_dict(torch.load('drive/My Drive/ColabNotebooks/Kolab/ML/workmodel/sh.pt'))
pred_list = []

names_list = []

for images, image_names in testloader:

    with torch.no_grad():

        images = images.to(device)

        output1 = model1(images)

        output2 = model3(images)

        output = torch.max(output1,output2)

        pred = F.softmax(output)

        pred = torch.argmax(pred, dim=1).cpu().numpy()

        pred_list += [p.item() for p in pred]

        names_list += [name for name in image_names]

        print(image_names)





sample_submission.image_name = names_list

sample_submission.label = pred_list

sample_submission.to_csv('drive/My Drive/ColabNotebooks/Kolab/ML/submission.csv', index=False)