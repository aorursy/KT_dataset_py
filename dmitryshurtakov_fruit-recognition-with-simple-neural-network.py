# импортируем необходимые библиотеки

import os

import shutil

import torch

import torchvision

import collections

import numpy as np

import matplotlib.pyplot as plt

import torch.nn as nn

import torch.nn.functional as F

from torchvision.datasets import ImageFolder

from torchvision.transforms import ToTensor, Resize, Compose

from torchvision.utils import make_grid

from torch.utils.data.dataloader import DataLoader

from torch.utils.data import random_split

%matplotlib inline

# создаём новые папки для переноса нужных файлов

directoryies = ['Apple', 'Carambola', 'Pomegranate', 'Pear', 'Plum', 'Tomatoes', 'Muskmelon']

parent_dir = '/kaggle/working'

for directory in directoryies:

    path = os.path.join(parent_dir, directory)

    os.mkdir(path)

    

path = '/kaggle/working/'

print(os.listdir(path))
# переносим нужные файлы в созданные директории

sources = ['../input/fruit-recognition/Apple/Total Number of Apples', 

           '../input/fruit-recognition/Carambola', 

           '../input/fruit-recognition/Pomegranate',

           '../input/fruit-recognition/Pear',

           '../input/fruit-recognition/Plum',

           '../input/fruit-recognition/Tomatoes', 

           '../input/fruit-recognition/muskmelon']



destinations = ['../output/kaggle/working/Apple',

                '../output/kaggle/working/Carambola', 

                '../output/kaggle/working/Pomegranate', 

                '../output/kaggle/working/Pear',

                '../output/kaggle/working/Plum',

                '../output/kaggle/working/Tomatoes', 

                '../output/kaggle/working/Muskmelon']



for i in range(len(sources)):

    destination = shutil.copytree(sources[i], destinations[i]) 
# посмотрим на получившиеся из дирекотрий классы

data_dir = '../output/kaggle/working'

classes = os.listdir(data_dir)

print(classes)
# приведём к одном размеру изначальные изображения и трансформируем их в тензоры

dsize = (248, 248)

composed = Compose([Resize(dsize), ToTensor()])



dataset = ImageFolder(data_dir, composed)

len(dataset)
# посчитаем количество элементов в каждом классе

class_count = collections.Counter(classes[label] for img, label in dataset)

class_count
# проверим количество классов

len(classes)
# посмотрим на единичный элемент в датасете

img, label = dataset[5]

print(img.shape, label)

img
# определим функцию для отображения одного элемента 

def show_example(img, label):

    print('Label: ', dataset.classes[label], "("+str(label)+")")

    plt.imshow(img.permute(1, 2, 0))

    

show_example(*dataset[5])
# создадим тренировочную и валидационную выборки из изначального датасета

torch.manual_seed(43)



val_size = 1500

test_size = 3000

train_size = len(dataset) - val_size - test_size



train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

len(train_ds), len(val_ds), len(test_ds)
# зададим размер батча и создадим загрузчики получившихся выборок с помощью DataLoader

batch_size = 32

train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)

val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)

test_loader = DataLoader(test_ds, batch_size*2, num_workers=4, pin_memory=True)
# отобразим один батч

for images, _ in train_loader:

    print('images.shape:', images.shape)

    plt.figure(figsize=(16,8))

    plt.axis('off')

    plt.imshow(make_grid(images, nrow=8).permute((1, 2, 0)))

    break
# создадим простую функцию для оценки точности

def accuracy(outputs, labels):

    _, preds = torch.max(outputs, dim=1)

    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
# создадим базовый класс модели (без архитектуры)

class ImageClassificationBase(nn.Module):

    def training_step(self, batch):

        images, labels = batch 

        out = self(images)                  

        loss = F.cross_entropy(out, labels) 

        return loss

    

    def validation_step(self, batch):

        images, labels = batch 

        out = self(images)                    

        loss = F.cross_entropy(out, labels)   

        acc = accuracy(out, labels)           

        return {'val_loss': loss.detach(), 'val_acc': acc}

        

    def validation_epoch_end(self, outputs):

        batch_losses = [x['val_loss'] for x in outputs]

        epoch_loss = torch.stack(batch_losses).mean()   

        batch_accs = [x['val_acc'] for x in outputs]

        epoch_acc = torch.stack(batch_accs).mean()      

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    

    def epoch_end(self, epoch, result):

        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
# создадим функции обучения и оценки

def evaluate(model, val_loader):

    outputs = [model.validation_step(batch) for batch in val_loader]

    return model.validation_epoch_end(outputs)



def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):

    history = []

    optimizer = opt_func(model.parameters(), lr)

    for epoch in range(epochs):

        # Training Phase 

        for batch in train_loader:

            loss = model.training_step(batch)

            loss.backward()

            optimizer.step()

            optimizer.zero_grad()

        # Validation phase

        result = evaluate(model, val_loader)

        model.epoch_end(epoch, result)

        history.append(result)

    return history
# проверим доступность GPU

torch.cuda.is_available()
# создадим функцию для получения доступа к GPU (если не доступен, то к CPU)

def get_default_device():

    if torch.cuda.is_available():

        return torch.device('cuda')

    else:

        return torch.device('cpu')



device = get_default_device()

device
# создадим функцию для переноса данных на GPU

def to_device(data, device):

    if isinstance(data, (list,tuple)):

        return [to_device(x, device) for x in data]

    return data.to(device, non_blocking=True)



class DeviceDataLoader():

    

    def __init__(self, dl, device):

        self.dl = dl

        self.device = device

        

    def __iter__(self):

        for b in self.dl: 

            yield to_device(b, self.device)



    def __len__(self):

        return len(self.dl)
# функции для отоборажения ошибки и точности модели

def plot_losses(history):

    losses = [x['val_loss'] for x in history]

    plt.plot(losses, '-x')

    plt.xlabel('epoch')

    plt.ylabel('loss')

    plt.title('Loss vs. No. of epochs')



def plot_accuracies(history):

    accuracies = [x['val_acc'] for x in history]

    plt.plot(accuracies, '-x')

    plt.xlabel('epoch')

    plt.ylabel('accuracy')

    plt.title('Accuracy vs. No. of epochs')
# загружаем данные на GPU

train_loader = DeviceDataLoader(train_loader, device)

val_loader = DeviceDataLoader(val_loader, device)

test_loader = DeviceDataLoader(test_loader, device)
# определяем размера входа (размер тензора изображения) и выхода (количество классов)

input_size = 3*248*248

output_size = 7
# расширяем первоночальную модель, добавляя архитектуру нейорнной сети

class FruitRecognitionModel(ImageClassificationBase):

    def __init__(self, in_size, out_size):

        super().__init__()

        self.linear1 = nn.Linear(in_size, 128)

        self.linear2 = nn.Linear(128, 32)

        self.linear3 = nn.Linear(32, out_size)

        

    def forward(self, xb):

        # Flatten images into vectors

        out = xb.view(xb.size(0), -1)

        # Apply layers & activation functions

        out = self.linear1(out)

        out = F.relu(out)

        out = self.linear2(out)

        out = F.relu(out)

        out = self.linear3(out)

        return out
# переносим модель на GPU

model = to_device(FruitRecognitionModel(input_size, out_size=output_size), device)

to_device(model, device)
# проверим точность и потери на валидационной выборке

history = [evaluate(model, val_loader)]

history
%%time

# обучаем модель, подобрав опытным путём оптимальный шаг и количество эпох

history += fit(20, 0.01, model, train_loader, val_loader)
# построим график потерь

plot_losses(history)
# построим график точности

plot_accuracies(history)
# проверим модель на тестовой выборке

evaluate(model, test_loader)
# сохраняем модель

torch.save(model.state_dict(), 'fruit-recognition-model.pth')