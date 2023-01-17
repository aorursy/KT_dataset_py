#Integrantes:

#Anderson Loureiro de Almeida  150153

#Artur Vinicius Oliveira       160796

#Giovanna da Cunha Correia     160763

#Ian de Melo                   160820

#Matheus Takeshi Hashimoto     160802
#Libraries

import torch

from torchvision import transforms, datasets, models

from torch import optim, cuda

from torch.utils.data import DataLoader, sampler, random_split

import torch.nn as nn

import torch.nn.functional as F

from PIL import Image

import numpy as np

import pandas as pd

import os

import seaborn as sns

import torchvision

import torchvision.transforms as transforms

import torch.optim as optim



torch.manual_seed(0)
print(os.listdir('../input/datasetdois')) #mostrar imagens ou pastas - depende do caminho
#Visualising Data

classes = []

img_classes = []

n_image = []

height = []

width = []

dim = []



# Using folder names to identify classes

for folder in os.listdir('../input/datasetdois'):#colocar caminho da raiz

    classes.append(folder)

    

    # Number of each image

    images = os.listdir('../input/datasetdois/'+folder)#colocar caminho da raiz

    n_image.append(len(images))

      

    for i in images:

        img_classes.append(folder)

        img = np.array(Image.open('../input/datasetdois/'+folder+'/'+i))#colocar caminho da raiz

        height.append(img.shape[0])

        width.append(img.shape[1])

    dim.append(img.shape[2])

    

df = pd.DataFrame({

    'classes': classes,

    'number': n_image,

    "dim": dim

})

print("Random heights:" + str(height[10]), str(height[123]))

print("Random Widths:" + str(width[10]), str(width[123]))

df
image_transforms = {

    # Train uses data augmentation

    'train':

    transforms.Compose([

        transforms.RandomResizedCrop(size=256, scale=(0.95, 1.0)),

        transforms.RandomRotation(degrees=15),

        transforms.ColorJitter(),

        transforms.RandomHorizontalFlip(),

        transforms.CenterCrop(size=224),  # Image net standards

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406],

                             [0.229, 0.224, 0.225])  # Imagenet standards

    ]),

    'val':

    transforms.Compose([

        transforms.Resize(size=256),

        transforms.CenterCrop(size=224),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),

    'test':

    transforms.Compose([

        transforms.Resize(size=256),

        transforms.CenterCrop(size=224),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),

}
batch_size = 20



all_data = datasets.ImageFolder(root='../input/datasetdois')

train_data_len = int(len(all_data)*0.8)

valid_data_len = int((len(all_data) - train_data_len)/2)

test_data_len = int(len(all_data) - train_data_len - valid_data_len)

train_data, val_data, test_data = random_split(all_data, [train_data_len, valid_data_len, test_data_len])

train_data.dataset.transform = image_transforms['train']

val_data.dataset.transform = image_transforms['val']

test_data.dataset.transform = image_transforms['test']

print(len(train_data), len(val_data), len(test_data))



train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
#TESTE 1 +/- 85%

class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5)

        self.conv2_bn = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(16 * 53 * 53, 120)

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, 2)

       

    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = self.pool(x)

        x = F.relu(self.conv2(x))

        x = self.pool(x)

        x = self.conv2_bn(x)

        x = x.view(-1, 16 * 53 * 53)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

      

        return x





net = Net()

"""

Nesta estrutura manteve-se os parâmetros de convolução da estrutura desenvolvida pelo

professor e mudou-se somente o tamanho da imagem, como consequência obteve-se uma acurácia muito alta, porém um dos 

motivos da alta acurácia pode ser por conta da brusca diminuição na etapa de classificação e assim podendo 

fazer com que o código fique muito genérico.

"""
#TESTE 2 +/- 42%

class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5)

        self.conv2_bn = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(16 * 53 * 53, 5000)

        self.fc2 = nn.Linear(5000, 2500)

        self.fc3 = nn.Linear(2500, 1250)

        self.fc4 = nn.Linear(1250, 625)

        self.fc5 = nn.Linear(625, 300)

        self.fc6 = nn.Linear(300, 120)

        self.fc7 = nn.Linear(120, 84)

        self.fc8 = nn.Linear(84, 2)



    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = self.pool(x)

        x = F.relu(self.conv2(x))

        x = self.pool(x)

        x = self.conv2_bn(x)

        x = x.view(-1, 16 * 53 * 53)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))

        x = F.relu(self.fc4(x))

        x = F.relu(self.fc5(x))

        x = F.relu(self.fc6(x))

        x = F.relu(self.fc7(x))

        x = self.fc8(x)

        return x





net = Net()

"""

Nesta estrutura manteve-se a estrutura referente ao teste 1 e aumentou-se a quantidade de camadas ocultadas.

Ao aumentar o número de camadas o código fica mais complexo e portanto sua acurácia diminui. Uma quantidade 

alta de camadas para esta rede somente a deixa complexa e com valores muito abaixo de acurácia.

"""
#TESTE 3 +/- 42%

class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)

        self.conv2 = nn.Conv2d(6, 16, 5)

        self.conv3 = nn.Conv2d(16, 25, 5) 

        self.conv4 = nn.Conv2d(25, 32, 5)

        self.conv2_bn = nn.BatchNorm2d(16)

        self.conv3_bn = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(2, 2)

        self.pool2 = nn.MaxPool2d(4, 4)

        self.fc1 = nn.Linear(32 * 2 * 2, 86)

        self.fc2 = nn.Linear(86, 44)

        self.fc3 = nn.Linear(44, 2)



    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = self.pool(x)

        x = F.relu(self.conv2(x))

        x = self.pool(x)

        x = self.conv2_bn(x)

        x = F.relu(self.conv3(x))

        x = self.pool2(x)

        x = F.relu(self.conv4(x))

        x = self.pool2(x)

        x = self.conv3_bn(x)

        x = x.view(-1, 32 * 2 * 2)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x





net = Net()

"""

Nesta estrutura duplicou-se a quantidade de convoluções, sumarização e normalização, porém manteve-se

a mesma quantidade de camadas ocultas que o teste 1. Duplicar os parâmetros tras complexidade para a rede e como 

consequência teve sua acurácia menor em relação ao primeiro teste que obteve uma acurácia bem mais elevada.

"""
#TESTE 4 +/- 47% 

class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)

        self.pool = nn.MaxPool2d(6, 6)

        self.conv2 = nn.Conv2d(6, 16, 5)

        self.conv2_bn = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, 2)

       

    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = self.pool(x)

        x = F.relu(self.conv2(x))

        x = self.pool(x)

        x = self.conv2_bn(x)

        x = x.view(-1, 16 * 5 * 5)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

      

        return x





net = Net()

"""

Nesta estrutura alterou-se o tamanho da janela na sumarização. De 2 para 6 havendo uma diminuição na acuracia devido a

analise ser feita em areas maiores generalizando as informações da janela

"""
#TESTE 5 +/- 76%

class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.pool2 = nn.MaxPool2d(4, 4)

        self.conv2 = nn.Conv2d(6, 16, 5)

        self.conv2_bn = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(16 * 26 * 26, 5000)

        self.fc2 = nn.Linear(5000, 500)

        self.fc3 = nn.Linear(500, 2)

       

    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = self.pool(x)

        x = F.relu(self.conv2(x))

        x = self.pool2(x)

        x = self.conv2_bn(x)

        x = x.view(-1, 16 * 26 * 26)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

      

        return x





net = Net()

"""

Nesta estrutura intercalou-se dois valores diferentes de sumarização aumentando o valor de 2 

para o primeiro e 4 para o segundo. Na primeira sumarização foi possível obter características importantes da imagem

já na segundo houve uma maior generalização em relação a primeira e colocar essas duas em sequência garante uma melhor

característica da imagem.

"""


torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



net = Net()

net.to(device)



criterion = nn.CrossEntropyLoss().to(device)

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)



for epoch in range(2):  # loop over the dataset multiple times



    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):

        # get the inputs; data is a list of [inputs, labels]

        inputs, labels = data[0].to(device), data[1].to(device)



        # zero the parameter gradients

        optimizer.zero_grad()



        # forward + backward + optimize

        outputs = net(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()



        # print statistics

        running_loss += loss.item()

        if i % 5 == 4:    # print every 5 mini-batches

            print('[%d, %5d] loss: %.3f' %

                  (epoch + 1, i + 1, running_loss / 5))

            running_loss = 0.0



print('Finished Training')
correct = 0

total = 0



with torch.no_grad():

    for data in test_loader:

        images, labels = data[0].to(device), data[1].to(device)

        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()



print('Accuracy of the network on the 10000 test images: %d %%' % (

    100 * correct / total))