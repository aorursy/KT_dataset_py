# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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

from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader, random_split

import albumentations 

from albumentations.pytorch import ToTensorV2 as AT

#import torch_optimizer as optim





import matplotlib.pyplot as plt


#PATH = '/kaggle/input/'

train_path = "/kaggle/input/new-train/NEW DATA/"

test_path = "/kaggle/input/competition2/test/"

sample_submission = pd.read_csv("/kaggle/input/sample-submission/sample_submission.csv")

train_list = os.listdir(train_path)

test_list = os.listdir(test_path)

print(len(train_list), len(test_list))
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

        

        #label = 0

        

        if image_name.startswith('1'):

            self.label = 1        

        elif image_name.startswith('2'):

            self.label = 2

        elif image_name.startswith('3'):

            self.label = 3

        elif image_name.startswith('4'):

            self.label = 4

        elif image_name.startswith('5'):

            self.label = 5

        elif image_name.startswith('6'):

            self.label = 6

        elif image_name.startswith('7'):

            self.label = 7

        else:

            self.label = 0

            

        if self.transform:

            augmented = self.transform(image=image)

            image = augmented["image"]

        

        if self.mode == "train":

            return image, self.label

        else:

            return image, image_name
#зададим немного гиперпараметров

batch_size = 55

num_workers = 0

img_size = 256
data_transforms = albumentations.Compose([

    albumentations.Resize(img_size, img_size),

    albumentations.ShiftScaleRotate(rotate_limit=0), #rotate_limit=0

    albumentations.Normalize(),

    #albumentations.ChannelShuffle(always_apply=False, p=0.5),

    albumentations.RandomRotate90(always_apply=False),

    #albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, always_apply=False, p=0.2),

    AT()

    ])

# RandomRotate90(always_apply=False)

#



data_transforms_test = albumentations.Compose([

    albumentations.Resize(img_size, img_size),

    albumentations.Normalize(),

    AT()

    ])
#Инициализируем датасеты

trainset = ChartsDataset(train_path, train_list,  transform = data_transforms)

testset = ChartsDataset(test_path, test_list,  transform=data_transforms_test, mode="test")
#Разделим трейновую часть на трейн и валидацию. Попробуем другой способ.

valid_size = int(len(train_list) * 0.1)

train_set, valid_set = torch.utils.data.random_split(trainset, 

                                    (len(train_list)-valid_size, valid_size))
#создаем даталоадеры для всех 3х подвыборок.

trainloader = torch.utils.data.DataLoader(train_set, pin_memory=True, 

                                        batch_size=batch_size, shuffle=True)



validloader = torch.utils.data.DataLoader(valid_set, pin_memory=True, 

                                        batch_size=batch_size, shuffle=True)



testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size,

                                         num_workers = num_workers)
samples, labels = next(iter(trainloader))

plt.figure(figsize=(16,24))

grid_imgs = torchvision.utils.make_grid(samples[:32])

np_grid_imgs = grid_imgs.numpy()

print(labels)

plt.imshow(np.transpose(np_grid_imgs, (1,2,0)))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device
#не забудьте включить интернет в правом меню настроек ---------------------------------------------------------------------->

#model = torchvision.models.resnet18(pretrained=True, progress=True)

model = torchvision.models.resnext50_32x4d(pretrained=True, progress=True)

#тут пример как заморозить все слои, как поступить вам решайте сами, от этого тоже много зависит)



for param in model.parameters():

    param.requires_grad = True

    

#in_features = model.fc.in_features

#model.avgpool = nn.AdaptiveAvgPool2d(output_size=1)

#model.maxpool = nn.MaxPool2d(output_size=1)

#model.fc = nn.Linear(in_features, 8)

#model.classifier._modules['6'] = nn.Linear(4096, 8)



for name, child in model.named_children():

    print(name)
for name, child in model.named_children():

   if name in ['layer4','fc', 'avgpool']:

       print(name + ' is unfrozen')

       for param in child.parameters():

           param.requires_grad = True

   else:

       print(name + ' is frozen')

       for param in child.parameters():

           param.requires_grad = False
in_features = model.fc.in_features

model.avgpool = nn.AdaptiveAvgPool2d(output_size=1)

#model.maxpool = nn.MaxPool2d(output_size=1)

model.fc = nn.Linear(in_features, 8)

#model.classifier._modules['6'] = nn.Linear(4096, 8)
def train_model(model_conv, train_loader, valid_loader, criterion, optimizer, sheduler, n_epochs):

    model_conv.to(device)

    valid_loss_min = np.Inf

    val_loss_min = np.Inf

    patience = 12

    # сколько эпох ждем до отключения

    p = 0

    # иначе останавливаем обучение

    stop = False



    # количество эпох

    for epoch in range(1, n_epochs+1):

        print(time.ctime(), 'Epoch:', epoch)



        train_loss = []



        for batch_i, (data, target) in enumerate(train_loader): # in enumerate(tqdm(train_loader)):

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

        correct = 0

        for batch_i, (data, target) in enumerate(valid_loader):

            data, target = data.to(device), target.to(device)

            output = model_conv(data)

            _, predicted = torch.max(output.data, 1)

            correct += (predicted == target).sum().item()

            loss = criterion(output, target)

            val_loss.append(loss.item()) 



            

        acc = correct / len(valid_set)

        print('{:.6f}% Accuracy'.format(acc*100))





        print(f'Epoch {epoch}, train loss: {np.mean(train_loss):.4f}, valid loss: {np.mean(val_loss):.4f}.')



        valid_loss = np.mean(val_loss)

        tr_loss = np.mean(train_loss)

        scheduler.step(valid_loss)

        #print('Learning rate: {:.8f}'.format(scheduler.get_lr()[0]))

        if valid_loss <= valid_loss_min:

            print('Validation loss decreased ({:.6f} --> {:.6f}).'.format(

            valid_loss_min,

            valid_loss))

            valid_loss_min = valid_loss

            p = 0

        

        if val_loss[epoch] < val_loss_min:

            print('Current val loss is smaller now {:.6f} --> {:.6f}. Saving model.'.format(val_loss_min, val_loss[epoch]))

            val_loss_min = val_loss[epoch]

            torch.save(model_conv.state_dict(), 'model.pt')

            

        

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
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

#optimizer = optim.SGDW(model.parameters(),lr=0.05, momentum=0.9, dampening=0, weight_decay=0.001,  nesterov=True,)

#optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

#optimizer = torch.optim.Adamax(model.parameters(), lr=0.0005)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=3)

    

#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0, last_epoch=-1)
model_resnet, train_loss, val_loss = train_model(model, trainloader, validloader, criterion, 

                              optimizer, scheduler, n_epochs = 100)
model.load_state_dict(torch.load('/kaggle/working/model.pt'))

model.to(device)

model.eval()

pred_list = []

names_list = []

poss0_list = []

poss1_list = []

poss2_list = []

poss3_list = []

poss4_list = []

poss5_list = []

poss6_list = []

poss7_list = []

for images, image_names in testloader:

    with torch.no_grad():

        images = images.to(device)

        output = model(images)

        print(output)

        pred = F.softmax(output)

        print(pred)

        poss0_list += [p[0] for p in pred.cpu().numpy()]

        poss1_list += [p[1] for p in pred.cpu().numpy()]

        poss2_list += [p[2] for p in pred.cpu().numpy()]

        poss3_list += [p[3] for p in pred.cpu().numpy()]

        poss4_list += [p[4] for p in pred.cpu().numpy()]

        poss5_list += [p[5] for p in pred.cpu().numpy()]

        poss6_list += [p[6] for p in pred.cpu().numpy()]

        poss7_list += [p[7] for p in pred.cpu().numpy()]

        pred1 = torch.argmax(pred, dim=1).cpu().numpy()

        print(pred1)

        pred_list += [p.item() for p in pred1]

        names_list += [name for name in image_names]





sample_submission.image_name = names_list

sample_submission.label = pred_list

sample_submission.to_csv('submissionresnext.csv', index=False)

sample_submission["0"] = poss0_list

sample_submission["1"] = poss1_list

sample_submission["2"] = poss2_list

sample_submission["3"] = poss3_list

sample_submission["4"] = poss4_list

sample_submission["5"] = poss5_list

sample_submission["6"] = poss6_list

sample_submission["7"] = poss7_list

sample_submission.to_csv('submission_probresnext.csv', index=False)
model.load_state_dict(torch.load('/kaggle/working/model.pt'))

model.to(device)

model.eval()

correct = 0



with torch.no_grad():

  for data, target in validloader:

    data = data.to(device=device)

    target = target.to(device=device)

    outputs = model(data)

    _, predicted = torch.max(outputs.data, 1)

    correct += (predicted == target).sum().item()





acc = correct / len(valid_set)

print(acc)