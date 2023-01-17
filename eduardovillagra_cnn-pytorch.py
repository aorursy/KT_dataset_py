import torch  # 	El paquete PyTorch de nivel superior y la biblioteca tensorial.

import torch.nn as nn   #	Un subpaquete que contiene módulos y clases extensibles para construir redes neuronales.

import torch.optim as optim # Un subpaquete que contiene operaciones de optimización estándar como SGD y Adam.

import torch.nn.functional as F # Una interfaz funcional que contiene operaciones típicas utilizadas para construir redes neuronales como funciones de pérdida y convoluciones.

import torchvision # Un paquete que proporciona acceso a conjuntos de datos populares, arquitecturas de modelos y transformaciones de imágenes para la visión por computadora.

import torchvision.transforms as transforms # Una interfaz que contiene transformaciones comunes para el procesamiento de imágenes.

import torchvision.datasets as dset

from torch.utils import data as D



# Otras importaciones



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from collections import Counter



from sklearn.metrics import confusion_matrix, accuracy_score

#from plotcm import plot_confusion_matrix



import pdb

torch.set_printoptions(linewidth=120)
transform = transforms.Compose(

    [transforms.Resize((60,60)),

     transforms.ToTensor(),

     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



data = dset.ImageFolder(root="../input/the-simpsons-characters-dataset/simpsons_dataset/simpsons_dataset",transform=transform)

dataloader = torch.utils.data.DataLoader(data, batch_size=16,shuffle=True,num_workers=2)



dataiter = iter(dataloader)

images, labels = dataiter.next()
type(data)
type(trainset)
data.len=len(data)

train_len = int(0.7*data.len)

test_len = data.len - train_len

trainset, testset = D.random_split(data, lengths=[train_len, test_len])
def imshow(img):

    img = img / 2 + 0.5     # unnormalize

    npimg = img.numpy()

    plt.figure(figsize=(15,15))

    plt.imshow(np.transpose(npimg, (1, 2, 0)))



imshow(torchvision.utils.make_grid(images))
class Network(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)

        

        self.fc1 = nn.Linear(in_features=32 * 15 * 15, out_features=200)

        self.fc2 = nn.Linear(in_features=200, out_features=90)

        self.out = nn.Linear(in_features=90, out_features=42)

        

    def forward(self, t):

        # (1) input layer

        t = t

        # (2) hidden conv layer

        t = self.conv1(t)

        t = F.relu(t)

        t = F.max_pool2d(t, kernel_size=2,stride=2)



        # (3) hidden conv layer

        t = self.conv2(t)

        t = F.relu(t)

        t = F.max_pool2d(t, kernel_size=2,stride=2)



        # (4) hidden linear layer

        t = t.reshape(-1, 32 * 15 * 15)

        t = self.fc1(t)

        t = F.relu(t)



        # (5) hidden linear layer

        t = self.fc2(t)

        t = F.relu(t)



        # (6) output layer

        t = self.out(t)

        #t = F.softmax(t, dim=1)



        return t
#Funcion predicciones 

def get_num_correct(preds, labels):

    return preds.argmax(dim=1).eq(labels).sum().item()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

network = Network()

network.to(device)

optimizer = optim.Adam(network.parameters(), lr=0.001)

train_loader = torch.utils.data.DataLoader(

    trainset

    ,batch_size=500

    ,shuffle=True

)



for epoch in range(20):



    total_loss = 0

    total_correct = 0



    for batch in train_loader: # Get Batch

        images, labels = batch 

        images = images.to(device)

        labels = labels.to(device)



        preds = network(images) # Pass Batch

        loss = F.cross_entropy(preds, labels) # Calculate Loss



        optimizer.zero_grad()

        loss.backward() # Calculate Gradients

        optimizer.step() # Update Weights



        total_loss += loss.item()

        total_correct += get_num_correct(preds, labels)



    print("epoch", epoch, "total_correct:", total_correct, "loss:", total_loss)
trainloader = torch.utils.data.DataLoader(trainset, 

                                         batch_size=500,

                                         shuffle=False)



total_correct = 0

total_images = 0

confusion_matrix = np.zeros([42,42], int)

with torch.no_grad():

    for data in trainloader:

        images, labels = data

        images = images.to(device)

        labels = labels.to(device)

        outputs = network(images)

        _, predicted = torch.max(outputs.data, 1)

        total_images += labels.size(0)

        total_correct += (predicted == labels).sum().item()

        for i, l in enumerate(labels):

            confusion_matrix[l.item(), predicted[i].item()] += 1 



model_accuracy = total_correct / total_images * 100

print('Model accuracy on {0} train images: {1:.2f}%'.format(total_images, model_accuracy))
testloader = torch.utils.data.DataLoader(testset, 

                                         batch_size=500,

                                         shuffle=False)



total_correct = 0

total_images = 0

confusion_matrix = np.zeros([42,42], int)

with torch.no_grad():

    for data in testloader:

        images, labels = data

        images = images.to(device)

        labels = labels.to(device)

        outputs = network(images)

        _, predicted = torch.max(outputs.data, 1)

        total_images += labels.size(0)

        total_correct += (predicted == labels).sum().item()

        for i, l in enumerate(labels):

            confusion_matrix[l.item(), predicted[i].item()] += 1 



model_accuracy = total_correct / total_images * 100

print('Model accuracy on {0} test images: {1:.2f}%'.format(total_images, model_accuracy))