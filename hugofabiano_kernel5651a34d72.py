from torchvision import transforms, datasets, models

import torch

torch.manual_seed(123)

from torch import optim, cuda

from torch.utils.data import DataLoader, sampler, random_split

import torch.nn as nn



from PIL import Image

import numpy as np

import pandas as pd

import os

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

print(os.listdir('../input/cachorroelefante'))
#Nessa parte do codigo vamos fazer um pequeno tratamento nos arquivos



classes = []

img_classes = []

n_image = []

height = []

width = []

dim = []





# Aqui o programa separa os arquivos nas duas classes, que são definidas atraves dos nomes das pastas



for folder in os.listdir('../input/cachorroelefante'):

    classes.append(folder)

    

    #Aqui o programa numera as imagens tanto fazendo a contagem, como para identificalas

    

    images = os.listdir('../input/cachorroelefante/'+folder)

    n_image.append(len(images))

      

    for i in images:

        img_classes.append(folder)

        img = np.array(Image.open('../input/cachorroelefante/'+folder+'/'+i))

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
image_df = pd.DataFrame({

    "classes": img_classes,

    "height": height,

    "width": width

})

img_df = image_df.groupby("classes").describe()

print(img_df)
#Abrindo uma imagem apenas para se certificar que o importe e acesso aos dados esta dando certo

Image.open('../input/cachorroelefante/cachorro/a-new-puppy-2-1387331.jpg')
image_transforms = {

    #Nesse parte o programa esta fazendo o treinmento do uso de dados, e redimencionando as imagens para que elas fiquem em um formato igual e ideal para que o programa consiga ler

    'train':

    transforms.Compose([

        transforms.RandomResizedCrop(size=256, scale=(0.95, 1.0)),

        transforms.RandomRotation(degrees=15),

        transforms.ColorJitter(),

        transforms.RandomHorizontalFlip(),

        transforms.CenterCrop(size=224),  #Tamanho definido como para as imagens

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406],

                             [0.229, 0.224, 0.225])  #Padrão de imagem

    ]),

    'val':

    transforms.Compose([

        transforms.Resize(size=256),

        transforms.CenterCrop(size=224), #Tamanho definido como para as imagens

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),

    'test':

    transforms.Compose([

        transforms.Resize(size=256),

        transforms.CenterCrop(size=224), #Tamanho definido como para as imagens

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])   #Padrão de imagem

    ]),

}
def imshow_tensor(image, ax=None, title=None):



    if ax is None:

        fig, ax = plt.subplots()



    #Definindo o canal de cores como a terceira dimensão

    

    image = image.numpy().transpose((1, 2, 0))



    #Invertendo as etapas de pré-processamento

    

    mean = np.array([0.485, 0.456, 0.406])

    std = np.array([0.229, 0.224, 0.225])

    image = std * image + mean



    #Cortando os valores de pixel da imagem

    

    image = np.clip(image, 0, 1)



    ax.imshow(image)

    plt.axis('off')



    return ax, image
img = Image.open('../input/cachorroelefante/cachorro/a-new-puppy-2-1387331.jpg')

img
transform = image_transforms['train']

plt.figure(figsize=(24, 24))



for i in range(16):

    ax = plt.subplot(4, 4, i + 1)

    _ = imshow_tensor(transform(img), ax=ax)



plt.tight_layout()
batch_size = 128



all_data = datasets.ImageFolder(root='../input/cachorroelefante')

train_data_len = int(len(all_data)*0.86) #!

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
trainiter = iter(train_loader)

features, labels = next(trainiter)

print(features.shape, labels.shape)
def train(model,

         criterion,

         optimizer,

         train_loader,

         val_loader,

         save_location,

         early_stop=3,

         n_epochs=20,

         print_every=2):

   

 #Inicializando algumas variáveis

  valid_loss_min = np.Inf

  stop_count = 0

  valid_max_acc = 0

  history = []

  model.epochs = 0

  




import torch.nn as nn

import torch.nn.functional as F





class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)      #CONVOLUÇÃO

        self.pool = nn.MaxPool2d(2, 2)       #POOLING

        self.conv2 = nn.Conv2d(6, 20, 5)     #NORMALIZAÇÃO

        self.conv2_bn = nn.BatchNorm2d(20)

        self.fc1 = nn.Linear(20 * 53 * 53, 5000)

        self.fc2 = nn.Linear(5000, 120)#(16*53*53, a)

        self.fc3 = nn.Linear(120, 2) #(a,2)



    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = self.pool(x)

        x = F.relu(self.conv2(x))

        x = self.pool(x)

        x = self.conv2_bn(x)

        x = x.view(-1, 20 * 53 * 53)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x





net = Net()

#Movendo os dados para GPU e executando novamente o treinamento

import torch.optim as optim



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



net = Net()

net.to(device)



criterion = nn.CrossEntropyLoss().to(device)

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)





#loop sobre o conjunto de dados várias vezes

for epoch in range(10):  



    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):

        #obter as entradas; data é uma lista de [entradas, etiquetas]

        

        inputs, labels = data[0].to(device), data[1].to(device)

        #inputs, labels = data



        #zerando os gradientes dos parâmetros

        optimizer.zero_grad()

        

        # forward + backward + optimize

        outputs = net(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()



        

        running_loss += loss.item()

        if i % 2000 == 1999:    

            print('[%d, %5d] loss: %.3f' %

                  (epoch + 1, i + 1, running_loss / 2000))

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



print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))