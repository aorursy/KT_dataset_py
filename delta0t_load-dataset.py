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
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import copy
import tqdm
from PIL import Image

import albumentations
from albumentations import pytorch as AT



%matplotlib inline
train_dir = '/kaggle/input/korpus-ml-2/train/train/train/'
test_dir = '/kaggle/input/korpus-ml-2/test/test/test/'
train_files = os.listdir(train_dir)
test_files = os.listdir(test_dir)
print(len(train_files), len(test_files))
train_files[:10]
class PrepareDataset(Dataset):

    def __init__(self, file_list, dir, transform=None, mode='train'):
        self.file_list = file_list
        self.dir = dir
        self.mode = mode
        self.transform = transform
        self.label = 0
        self.name_to_label = {'bar_chart' : 1, 'diagram' : 2, 'flow_chart' : 3,
                              'graph' : 4, 'growth_chart' : 5, 'pie_chart' : 6,
                              'table' : 7, 'just_image' : 0}
        self.label_to_name = {1 : 'bar_chart', 2 : 'diagram', 3 : 'flow_chart',
                              4 : 'graph', 5 : 'growth_chart', 6 : 'pie_chart',
                              7 : 'table', 0 : 'just_image'}
            
    def __len__(self):
        return len(self.file_list)
    
    #метод который позволяет нам индексировать датасет
    def __getitem__(self, idx):
        #считываем изображение
        image_name = self.file_list[idx]
        full_path = os.path.join(self.dir, image_name)
        if image_name.split(".")[1] == "gif":
          gif = cv2.VideoCapture(full_path)
          _, image = gif.read()
        else:
            image = cv2.imread(full_path)
 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.mode == 'train':
            for name, label in self.name_to_label.items():
                if name in self.file_list[idx]:
                    self.label = label
                    break

        #применяем аугментации
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        if self.mode == 'train':
            return image, self.label
        else:
            return image, image_name
!lscpu
import multiprocessing

multiprocessing.cpu_count()
#зададим немного гиперпараметров


batch_size = 32
num_workers = multiprocessing.cpu_count()
img_size = 256
#Пример того как выглядит обычно набор аугментаций. 
#В зависимости от задачи может отличаться количеством и сложностью.

data_transforms = albumentations.Compose([
    albumentations.Resize(img_size, img_size),
#     albumentations.Normalize(),
    AT.ToTensor()
    ])

#обычно аугментации для трейн и тест датасетов разделают. 
#На тесте обычно не нужно сильно изменять изображения
data_transforms_test = albumentations.Compose([
    albumentations.Resize(img_size, img_size),
#     albumentations.Normalize(),
    AT.ToTensor()
    ])
#Инициализируем датасеты
trainset = PrepareDataset(train_files, train_dir, transform = data_transforms)
testset = PrepareDataset(test_files, test_dir, 
                        transform=data_transforms_test, mode = "test")
#Разделим трейновую часть на трейн и валидацию.

valid_size = int(len(train_files) * 0.15) #размер валидационной части (10-15%)
indices = torch.randperm(len(trainset)) #создадим лист индексов
#определим подвыборки для трейн и валидации
train_indices = indices[:len(indices)-valid_size] 
valid_indices = indices[len(indices)-valid_size:]

#создаем даталоадеры для всех 3х подвыборок.
trainloader = torch.utils.data.DataLoader(trainset, pin_memory=True, 
                                        batch_size=batch_size,
                                        sampler=SubsetRandomSampler(train_indices))
validloader = torch.utils.data.DataLoader(trainset, pin_memory=True, 
                                        batch_size=batch_size,
                                        sampler=SubsetRandomSampler(valid_indices))

testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size,
                                         num_workers = num_workers)
#Проверим работоспособность
samples, labels = next(iter(validloader))
plt.figure(figsize=(32,48))
grid_imgs = torchvision.utils.make_grid(samples[:24])
np_grid_imgs = grid_imgs.numpy()
# in tensor, image is (batch, width, height), so you have to transpose it to (width, height, batch) in numpy to show it.
plt.imshow(np.transpose(np_grid_imgs, (1,2,0)), interpolation='nearest')

# Посмотрим на данные и убедимся, что метки соответствуют классам изображений
for label in labels[:24]:
    print(trainset.label_to_name.get(int(label)))