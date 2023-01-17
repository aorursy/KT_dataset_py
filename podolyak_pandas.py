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

from PIL import Image

from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader, random_split

import albumentations 

from albumentations.pytorch import ToTensorV2 as AT





import matplotlib.pyplot as plt
PATH = 'yandex/'

train_path=list()

for directory in os.listdir(PATH):

    train_path.append(os.path.join(PATH, directory))

    

test_path = ("test/")



train_list=list()

for directory in train_path:

    for pic in os.listdir(directory):

        train_list.append(directory+'/'+pic)



test_list=list()

for pic in os.listdir(test_path):

    test_list.append(test_path+pic)

print(len(train_list), len(test_list))
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

            image = cv2.imread(self.path + image_name)

            

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        

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

        else:

            label = 0 #just_image

            

        if self.transform:

            augmented = self.transform(image=image)

            image = augmented["image"]

        

        if self.mode == "train":

            return image, label

        else:

            return image, image_name
batch_size = 64

num_workers = os.cpu_count()

img_size = 256
data_transforms = albumentations.Compose([

    albumentations.Resize(img_size, img_size),

    albumentations.CLAHE(),

    albumentations.ChannelShuffle(),

    albumentations.Downscale(),

    albumentations.Cutout(),

    albumentations.ShiftScaleRotate(),

    albumentations.Normalize(),

    AT()

    ])





data_transforms_test = albumentations.Compose([

    albumentations.Resize(img_size, img_size),

    albumentations.Normalize(),

    AT()

    ])
#Инициализируем датасеты

trainset = ChartsDataset('./', train_list,  transform=data_transforms)

testset = ChartsDataset('./', test_list,  transform=data_transforms_test, mode="test")
valid_size = int(len(train_list) * 0.1)

train_set, valid_set = torch.utils.data.random_split(trainset, 

                                    (len(train_list)-valid_size, valid_size))
trainloader = torch.utils.data.DataLoader(train_set, pin_memory=True, 

                                        batch_size=batch_size, shuffle=True,

                                        num_workers = num_workers)



validloader = torch.utils.data.DataLoader(valid_set, pin_memory=True, 

                                        batch_size=batch_size, shuffle=True,

                                        num_workers = num_workers)



testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size,

                                         num_workers = num_workers)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device
model = torchvision.models.resnet152(pretrained=True, progress=True)
for param in model.parameters():

    param.requires_grad = False



in_features = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(in_features, 1024),

                         nn.Linear(1024,8))
def train_model(model_conv, train_loader, valid_loader, criterion, optimizer, sheduler, n_epochs):

    model_conv.to(device)

    valid_loss_min = np.Inf

    patience = 10

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



    return model_conv, train_loss, val_loss
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0007)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=3,)
model_resnet, train_loss, val_loss = train_model(model, trainloader, validloader, criterion, 

                              optimizer, scheduler, n_epochs=80,)
sample_submission = pd.read_csv("sample_submission.csv")

model.to(device)

model.eval()

pred_list = []

names_list = []

for images, image_names in testloader:

    with torch.no_grad():

        images = images.to(device)

        output = model(images)

        pred = F.softmax(output)

        pred = torch.argmax(pred, dim=1).cpu().numpy()

        pred_list += [p.item() for p in pred]

        names_list += [name for name in image_names]





sample_submission.image_name = names_list

sample_submission.label = pred_list

sample_submission["image_name"]=sample_submission["image_name"].apply(lambda x: x.split('/')[1])

sample_submission.to_csv('submission_152_10-3.csv', index=False)