# импорт библиотек

import numpy as np

import pandas as pd

import torch

import random

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

import os

import cv2

import time

import torchvision

from PIL import Image

from tqdm import tqdm_notebook as tqdm

from torch.utils.data import Dataset, DataLoader, random_split

import albumentations 

from albumentations.pytorch import ToTensorV2 as AT

import matplotlib.pyplot as plt
# определяем пути 

PATH = '/kaggle/input/korpus-ml-1/'

train_path = PATH + "train/train/"

test_path = PATH + "test/test/"

train_list = os.listdir(train_path)

test_list = os.listdir(test_path)

print(len(train_list), len(test_list))
# определяем свой датасет

class ChartsDataset(Dataset):

    

    def __init__(self, path, img_list, transform=None, mode='train'):

        self.path = path

        self.img_list = img_list

        self.transform=transform

        self.mode = mode

        

    def __len__(self):

        return len(self.img_list)

    

    def __getitem__(self, idx):

        image_name = self.img_list[idx]

        

        if image_name.split(".")[1] == "gif":

           gif = cv2.VideoCapture(self.path + image_name)

           _, image = gif.read()

        else:

            image = cv2.imread(self.path + image_name)

            

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        

        label = 0

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

            label = 0

            

        if self.transform:

            augmented = self.transform(image=image)

            image = augmented["image"]

        

        if self.mode == "train":

            return image, label

        else:

            return image, image_name
# зададим гиперпараметры

batch_size = 32

num_workers = 0

img_size = 256
# определим аугментации для тренировочной и тестовой выборки

data_transforms = albumentations.Compose([

    albumentations.Resize(img_size, img_size),

    albumentations.ShiftScaleRotate(),

    albumentations.Normalize(),

    AT()

    ])





data_transforms_test = albumentations.Compose([

    albumentations.Resize(img_size, img_size),

    albumentations.Normalize(),

    AT()

    ])
# инициализируем датасеты

trainset = ChartsDataset(train_path, train_list,  transform = data_transforms)

testset = ChartsDataset(test_path, test_list,  transform=data_transforms_test, mode="test")
# разделим трейновую часть на трейн и валидацию. Попробуем другой способ.

valid_size = int(len(train_list) * 0.1)

train_set, valid_set = torch.utils.data.random_split(trainset, 

                                    (len(train_list)-valid_size, valid_size))
# создаем даталоадеры для всех 3-х подвыборок.

trainloader = torch.utils.data.DataLoader(train_set, pin_memory=True, 

                                        batch_size=batch_size, shuffle=True)



validloader = torch.utils.data.DataLoader(valid_set, pin_memory=True, 

                                        batch_size=batch_size, shuffle=True)



testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size,

                                         num_workers = num_workers)
# посмотрим на данные

samples, labels = next(iter(trainloader))

plt.figure(figsize=(16,24))

grid_imgs = torchvision.utils.make_grid(samples[:32])

np_grid_imgs = grid_imgs.numpy()

print(labels)

plt.imshow(np.transpose(np_grid_imgs, (1,2,0)))
# определяем device

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device
# не забудьте включить интернет в правом меню настроек ---------------------------------------------------------------------->

model = torchvision.models.resnet18(pretrained=True, progress=True)



# пример заморозки слоев

for param in model.parameters():

    param.requires_grad = False

    

# настраиваем модель под свою задачу

in_features = model.fc.in_features

model.avgpool = nn.AdaptiveAvgPool2d(output_size=1)

model.fc = nn.Linear(in_features, 8)
# функция тренировки

def train_model(model_conv, train_loader, valid_loader, criterion, optimizer, sheduler, n_epochs):

    # переносим на GPU

    model_conv.to(device)

    

    valid_loss_min = np.Inf

    patience = 5

    # сколько эпох ждем до отключения

    p = 0

    # иначе останавливаем обучение

    stop = False



    # количество эпох

    for epoch in range(n_epochs):

        train_loss = []

        for batch_i, (data, target) in tqdm(enumerate(train_loader)):

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

        

        # проверяем, как дела на валидации        

        if valid_loss <= valid_loss_min:

            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))

            torch.save(model_conv.state_dict(), 'model.pt')

            valid_loss_min = valid_loss

            p = 0



        if valid_loss > valid_loss_min:

            p += 1

            print(f'{p} epochs of increasing val loss')

            # тормозим обучение

            if p > patience:

                print('Stopping training')

                stop = True

                break

                

        if stop:

            break

    return model_conv, train_loss, val_loss
# определяем лосс, оптимайзер, шедуллер

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=2,)
# запускаем обучение

model_resnet, train_loss, val_loss = train_model(model, trainloader, validloader, criterion, optimizer, scheduler, n_epochs=25,)
# тестируем модель

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





# сохраняем csv для сабмита

sample_submission.image_name = names_list

sample_submission.label = pred_list

sample_submission.to_csv('submission.csv', index=False)