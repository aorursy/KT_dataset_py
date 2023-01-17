#!pip install opencv-python

import numpy as np 

import pandas as pd 

import shutil

import os

import zipfile

import torch

import torch.nn as nn

import cv2 

import matplotlib.pyplot as plt

import torchvision

from torch.utils.data import Dataset, DataLoader

from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import transforms

import torch.nn.functional as F

import copy

from tqdm import tqdm_notebook as tqdm

import time

from PIL import Image

import random

import albumentations

from albumentations import pytorch as AT

from sklearn.metrics import accuracy_score

%matplotlib inline
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

seed_everything(40)  # 0.97 c 40
train_dir = '/kaggle/input/kkkkkk/All1/'

test_dir = '/kaggle/input/korpus-ml-1/test/test/'

train_files = os.listdir(train_dir)

test_files = os.listdir(test_dir)

submission = pd.read_csv("/kaggle/input/korpus-ml-1/sample_submission.csv")
print(len(train_files), len(test_files))
class DiagrammgDataset(Dataset):

    def __init__(self, file_list, dir, transform=None, tester=None):

        self.file_list = file_list

        self.dir = dir

        self.transform = transform

        self.tester = tester

          

    def __len__(self):

        return len(self.file_list)

    

    #метод который позволяет нам индексировать датасет

    def __getitem__(self, idx):

        #считываем изображение

        temp = self.dir+self.file_list[idx]

        if temp.split(".")[1] == 'gif':

            gif1 = cv2.VideoCapture(temp)   

            image = gif1.read()[1]

        else:

            image = cv2.imread(temp)



        if image is None:

            print(i, temp)

            labels = None

        else:

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 



            if 'bar_chart' in self.file_list[idx]:

                self.label = 1

            elif 'diagram' in self.file_list[idx]:

                self.label = 2

            elif 'flow_chart' in self.file_list[idx]:

                self.label = 3

            elif 'graph' in self.file_list[idx]:

                self.label = 4

            elif 'growth_chart' in self.file_list[idx]:

                self.label = 5

            elif 'pie_chart' in self.file_list[idx]:

                self.label = 6

            elif 'table' in self.file_list[idx]:

                self.label = 7

            elif 'just_image' in self.file_list[idx]:

                self.label = 0

            else:

                self.label = 0



            #применяем аугментации

            if self.transform:

                augmented = self.transform(image=image)

                image = augmented['image']

        if self.tester:

            return image, self.label, self.file_list[idx]

        else:

            return image, self.label



#зададим немного гиперпараметров

batch_size = 32

num_workers = 0

img_size = 256
#vgg19



data_transforms = albumentations.Compose([

    albumentations.Resize(256,256),

    albumentations.CenterCrop(224,224),

    albumentations.Normalize(),

    #albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, always_apply=False, p=0.5),

    #albumentations.GaussianBlur(),

    #albumentations.RandomContrast(),

    AT.ToTensor()

    ])



#обычно аугментации для трейн и тест датасетов разделают. 

#На тесте обычно не нужно сильно изменять изображения

data_transforms_test = albumentations.Compose([

    albumentations.Resize(256,256),

    albumentations.CenterCrop(224,224),

    #albumentations.GaussianBlur(),

    albumentations.Normalize(),

    AT.ToTensor()

    ])
#Инициализируем датасеты

trainset = DiagrammgDataset(train_files, train_dir, transform = data_transforms)

testset = DiagrammgDataset(test_files, test_dir, transform=data_transforms_test, tester = True)
#Разделим трейновую часть на трейн и валидацию. Попробуем другой способ.

valid_size = int(len(train_files) * 0.06)

train_set, valid_set = torch.utils.data.random_split(trainset, 

                                    (len(train_files)-valid_size, valid_size))
len(train_set), len(valid_set)
#создаем даталоадеры для всех 3х подвыборок.

trainloader = torch.utils.data.DataLoader(train_set, pin_memory=True, 

                                        batch_size=batch_size, shuffle=True)

validloader = torch.utils.data.DataLoader(valid_set, pin_memory=True, 

                                        batch_size=batch_size, shuffle=True)



testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size,

                                         num_workers = num_workers)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device
import torch.hub

model = torch.hub.load(

'pytorch/vision:v0.5.0', 

'vgg19_bn',

pretrained=True,)

#model = torchvision.models.resnet50(pretrained=True)

for param in model.parameters():

    param.requires_grad = True



#num_features = model.classifier[6].in_features

features = list(model.classifier.children())[:-1] # Remove last layer

features.extend([nn.Linear(4096, 8)]) # Add our layer with 4 outputs

model.classifier = nn.Sequential(*features) # Replace the model classifier



from sklearn.metrics import accuracy_score





def train_model(model_conv, train_loader, valid_loader, criterion, optimizer, scheduler, n_epochs):

    model_conv.to(device)

    valid_loss_min = np.Inf

    patience = 5

    # сколько эпох ждем до отключения

    p = 0

    # иначе останавливаем обучение

    stop = False



    # количество эпох

    for epoch in tqdm(range(1, n_epochs+1)):

        print(time.ctime(), 'Epoch:', epoch)



        train_loss = []

        pred_train_list = []

        pred_val_list = []

        target_train_list = []

        target_val_list = []

        for batch_i, (data, target) in enumerate(tqdm(train_loader)):



            data, target = data.to(device), target.to(device)



            optimizer.zero_grad()

            output = model_conv(data)

            for i, pred in enumerate(output):

                pred = torch.softmax(pred, dim = 0)

                pred = torch.argmax(pred)  

                pred = pred.cpu().detach().numpy()

                pred_train_list.append(pred)

                target_train_list.append(target[i].cpu().detach().numpy())

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

            for i, pred in enumerate(output):

                pred = torch.softmax(pred, dim = 0)

                pred = torch.argmax(pred)  

                pred_val_list.append(pred.cpu().detach().numpy())

                target_val_list.append(target[i].cpu().detach().numpy())

            loss = criterion(output, target)

            val_loss.append(loss.item()) 

        

        print(f'Epoch {epoch}, train loss: {np.mean(train_loss):.4f}, valid loss: {np.mean(val_loss):.4f}.')

        print("accuracy_train:",accuracy_score(target_train_list,pred_train_list))

        print("accuracy_val:",accuracy_score(target_val_list, pred_val_list)) 

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



        if stop:

            break

    return model_conv, train_loss, val_loss
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.0008,momentum=0.9) #    SGD(, lr=0.001, momentum=0.9)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=2,)
model_resnet, train_loss, val_loss = train_model(model, trainloader, validloader, criterion, optimizer, scheduler, n_epochs=40)
#torch.save(model_resnet, "modelVGG19BE(40,hue).pht")
'''

model.state_dict(torch.load('model.pt'))

'''

print("Lets infer on this model")
model_resnet.to(device)

model_resnet.eval()

pred_list = []

pred_list_name = []

th = 0

for images, imlbl, image_names in testloader:

    with torch.no_grad():

        images = images.to(device)

        output = model_resnet(images)

        for i, pred in enumerate(output):

            pred = torch.softmax(pred, dim = 0)

            pred = torch.argmax(pred)  

            pred = pred.cpu().detach().numpy()

            pred_list_name.append(image_names[i])

            pred_list.append(pred)

            th=th+1

drw = submission.image_name 

c = []

for i, item in enumerate(drw):

    c.append(item)
o = []

for i, item in enumerate(c):

    for k, ktem in enumerate(pred_list_name):

        if ktem == item:

            o.append(pred_list[k])

submission.label = o

submission.to_csv('sample_submissionOK2.csv', index=False)
