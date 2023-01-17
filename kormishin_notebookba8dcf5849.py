import numpy as np 

import pandas as pd

import torch

import os

import shutil

import matplotlib.pyplot as plt

%matplotlib inline
# создадим рабочие дирректории для картинок Muskmelon и Carambola

# перенесём картинки с Muskmelon и Carambola в рабочие дирректории



# фрукты, которые мы будем сравнивать друг с другом

fruits = ['Carambola', 'Muskmelon']



# общие пути для директорий

working_dir = '../output/kaggle/working/'

base_dir = '../input/fruit-recognition/'

        

def create_working_dirs_and_move_images():

    if(not os.path.isdir(base_dir)):

        print('Отсутствует исходная база с картинками: ', base_dir)

    

    # для каждого фрукта создаём новую рабочую директорию и перемещаем туда картинки

    for fruit in fruits:

        # в исходных путях имена директорий иногда встречаются с маленькой буквы

        if(fruit == 'Muskmelon'): 

            fruit_source = 'muskmelon'

        else:

            fruit_source = fruit

        

        # исходная и целевая директория для каждого фрукта

        fruit_source_path = os.path.join(base_dir, fruit_source)

        fruit_destination_path = os.path.join(working_dir, fruit)

        

        if(not os.path.isdir(fruit_source_path)):

            print('Отсутствует исходная база с картинками: ', fruit_source_path)

            break

        

        if(not os.path.isdir(fruit_destination_path)):

            # создаём рабочую диреткорию

            os.mkdir(os.path.join('/kaggle/working', fruit))

            # переещаем картинки

            shutil.copytree(fruit_source_path, fruit_destination_path)

            print('\nСоздана директория ', fruit_destination_path)

        else:

            print('\nРабочая директория ', fruit_destination_path, 'уже существует (картинки не будут повторно перемещаться)')

        

        # посчитаем число картинок для каждого фрукта в рабочей директории

        path, dirs, files = next(os.walk(fruit_destination_path))

        img_count = len(files)

        print(fruit, ': ', fruit_destination_path, ' - ', img_count, ' картинок')

        

    # отобразим все директории в рабочей папке

    print('\nВсе директории в рабочем пути:', os.listdir(working_dir))



create_working_dirs_and_move_images()
# посмотрим размерности всех картинок

from torchvision.datasets import ImageFolder

from torchvision.transforms import ToTensor, Compose, Grayscale



# трансформируем картинки в одноканальный формат (чёрно-белый в итоге) и в тензор

composed_transformers = Compose([Grayscale(num_output_channels=1), ToTensor()])

# composed_transformers = Compose([ToTensor()])

data = ImageFolder('../output/kaggle/working', composed_transformers)



# общий объём датасета

print('Всего картинок: ', len(data))



# перебор всех картинок с фруктами для подсчёта числа картинок с разными размерностями

for i in range(len(data)):

    img, label = data[i]

    if(i == 0):

        n = 1

        previous_img_shape = img.shape

        previous_label = label

        continue

    

    if( i == len(data)-1 ):

        n += 1

        print('Размерность для класса:', img.shape, " | Класс:", label, ' | Количество картинок:', n)

   

    if(previous_img_shape == img.shape and previous_label == label): 

        n += 1

        continue

    else:

        print('Размерность для класса:', previous_img_shape, ' | Класс:', previous_label, ' | Количество картинок:', n)

        n = 1

    

    previous_img_shape = img.shape

    previous_label = label
# функция для отображения картинки одного фрукта

def show_img(img, label):

    print('Фрукт:', data.classes[label], ' | Класс:', label)

    print(img.shape)

    # отобразим картинку предварительно убрав лишнюю размерность (размерность канала, т.к. он один)

    image_squeezed = torch.squeeze(img)

    plt.imshow(image_squeezed)



# отображаем случайную картинку (* - раскрывает результат в tuple)

show_img(*data[3])
# сформируем train и test выборки

from torch.utils.data import random_split



torch.manual_seed(42)



train_size = int(round(0.75*len(data)))

val_size = 0

test_size = len(data) - train_size



train, val, test = random_split(data, [train_size, val_size, test_size])
# отобразим один элемент из train

train[3]
# загрузчики выборок



# размер батча

batch_size = 8



train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4)

test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4)
# выведем батч с картинками (8 шт.) для выборки train

from torchvision.utils import make_grid



for images, _ in train_loader:

    print('images.shape:', images.shape)

    plt.figure(figsize=(16,8))

    plt.axis('off')

    plt.imshow(make_grid(images, nrow=8).permute((1, 2, 0)))

    break
# доступность GPU

# если мы на GPU, то будет выведено соответствующее уведомление, иначе - отобразится CPU

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if(not torch.cuda.is_available()):

    print('Текущий device - CPU. Необходимо включить GPU (видеокарта не подключена)')

else:

    print(device)
import torch.nn as nn

import torch.nn.functional as F



class neural(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(1*258*320, 128)

        self.fc2 = nn.Linear(128, 32)

        self.fc3 = nn.Linear(32, 2)



    def forward(self, x):

        # Flatten images into vectors

        x = x.view(x.size(0), -1)

        # Apply layers & activation functions

        x = F.sigmoid(self.fc1(x))

        x = F.sigmoid(self.fc2(x))

        x = self.fc3(x)

        return F.softmax(x, dim=1)



net = neural()

print(net)
# переносим сеть на GPU

net.to(device)
import torch.optim as optim



# переносим loss function на GPU

criterion = nn.CrossEntropyLoss().cuda()

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# переносим входные данные (inputs, labels = data) на GPU и тренируем сеть заново



# проходим через данные несколько раз 

# (2 эпохи для оптимизации скорости запуска в демонстрационных целях, в Production лучше использовать большее количество - 500, например)

for epoch in range(2):



    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):

        # получаем входные данные [inputs, labels]

        # inputs, labels = data

        inputs, labels = data[0].to(device), data[1].to(device)



        # обнуляем градиент

        optimizer.zero_grad()



        # forward + backward + optimize

        outputs = net(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()



        # статистика

        running_loss += loss.item()

        

        # печать каждые 100 мини-батчей

        if i % 100 == 99:

            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 100))

            running_loss = 0.0



print('Тренировка окончена')
# Сохраняем натренированную сеть для дальнейшего использования

PATH = './fruit_net.pth'

torch.save(net.state_dict(), PATH)
# отобразим 8 случайных картинок (количество определяется размером батча) из тестовой выборки

# расчёт происходит на CPU



# функция отображения картинки

def imshow(img):

    plt.figure(figsize=(16,8))

    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    plt.show()



# последовательно проходим по каждой картинке

dataiter = iter(test_loader)

images, labels = dataiter.next()



# отображаем картинки

imshow(make_grid(images))

# печатаем фактическую принадлежность к классу

classes = ('Carambola', 'Muskmelon')

print(' '.join('%5s' % classes[labels[j]] for j in range(8)))
# отобразим прогноз сети для тех же самых картинок



# загружаем сеть

net.load_state_dict(torch.load(PATH))

# переносим её на GPU

net.to(device)



# переносим imgages и labels на GPU

images, labels = images.to(device), labels.to(device)

# пропускаем картинки через сеть для прогноза

outputs = net(images)



# Выходом является инпульс для двух классов 

# Чем сильнее импульс, тем более вероятно сеть относит картинку к определённому классу

# Здесь мы получаем индекс максимального импульса

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(8)))
correct = 0

total = 0

with torch.no_grad():

    for data in test_loader:

        #images, labels = data

        images, labels = data[0].to(device), data[1].to(device)

        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()



print('Точность предсказания сети на всей тестовой выборке: %d %%' % (100 * correct / total))
# для тестовой выборки посмотрим точность предсказания каждого из фруктов

class_correct = list(0. for i in range(10))

class_total = list(0. for i in range(10))

with torch.no_grad():

    for data in test_loader:

        images, labels = data[0].to(device), data[1].to(device)

        outputs = net(images)

        _, predicted = torch.max(outputs, 1)

        c = (predicted == labels).squeeze()

        for i in range(4):

            label = labels[i]

            class_correct[label] += c[i].item()

            class_total[label] += 1





for i in range(2):

    print('Точность для %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))