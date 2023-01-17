import torch 

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from torchvision import datasets,transforms

import torch.nn.functional as F

from torch import nn,optim
#Generate Transform to Imgaes

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,)),])
#Load the train data and test data

trainset=datasets.FashionMNIST('~/.pytorch/FashionMNIST_data/',train=True,download=True,transform=transform)

testset=datasets.FashionMNIST('~/.pytorch/FashionMNIST_data/',train=False,download=True,transform=transform)
#Make a Data Loader for every trainset and testset

trainloader=torch.utils.data.DataLoader(trainset,batch_size=64,shuffle=True)

testloader=torch.utils.data.DataLoader(testset,batch_size=64,shuffle=True)
#Take batch of images and labels

images,labels=next(iter(trainloader))
#Imgaes shapes

print(images.shape)
#Encode the 10's labels of images

clothing = {0 : 'T-shirt/top',

            1 : 'Trouser',

            2 : 'Pullover',

            3 : 'Dress',

            4 : 'Coat',

            5 : 'Sandal',

            6 : 'Shirt',

            7 : 'Sneaker',

            8 : 'Bag',

            9 : 'Ankle boot'}
#Visualize and Explore data



fig = plt.figure()

for i in range(3):

  plt.subplot(2,3,i+1)

  plt.tight_layout()

  plt.imshow(images[i][0], cmap='gray', interpolation='none')

  plt.title("Ground Truth: {}".format(clothing[int(labels[i])]))

  plt.xticks([])

  plt.yticks([])

fig

class NeuralNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        #Creat 4 Hidden Layers

        self.fc1=nn.Linear(784,256)

        self.fc2=nn.Linear(256,128)

        self.fc3=nn.Linear(128,64)

        self.fc4=nn.Linear(64,10)

        

        #Dropout Layers

        self.dropout=nn.Dropout(p=0.5)

    def forward(self,x):

        #Flatten Images

        x=x.view(x.shape[0],-1)

        

        x=self.dropout(F.relu(self.fc1(x)))

        x=self.dropout(F.relu(self.fc2(x)))

        x=self.dropout(F.relu(self.fc3(x)))

        

        x=F.log_softmax(self.fc4(x),dim=1)

        

        return x

        
model=NeuralNetwork()

print(model)
optimizer=optim.Adam(model.parameters(),lr=0.0003)

crit=nn.NLLLoss()
#Train Data



epochs=30

step=0

train_losses,test_losses,accu=[],[],[]

for i in range(epochs):

    running_loss=0

    for images,labels in trainloader:

        optimizer.zero_grad()

        logps=model(images)

        loss=crit(logps,labels)

        loss.backward()

        optimizer.step()

        running_loss+=loss.item()

    else:

        test_loss=0

        accuracy=0

        

        #Turn off gradient for validation ,saves momery and computation

        

        with torch.no_grad():

            model.eval()

            for images,labels in testloader:

                logps=model(images)

                test_loss+=crit(logps,labels)

                ps=torch.exp(logps)

                topp,topc=ps.topk(1,dim=1)

                equal=topc==labels.view(*topc.shape)

                accuracy+=torch.mean(equal.type(torch.FloatTensor))

        model.train()

        

        train_losses.append(running_loss/len(trainloader))

        test_losses.append(test_loss/len(testloader))

        accu.append(accuracy)

        

        print("Epoch: {}/{}".format(i+1,epochs),

        "Training Loss {}".format(running_loss/len(trainloader)),

        "Testing Loss {}".format(test_loss/len(testloader)),

        "Accuracy {}".format(accuracy/len(testloader)))
%matplotlib inline

%config InlineBackend.figure_format='retina'

plt.plot(train_losses,label='Traning Loss')

plt.plot(test_losses,label='Validation Loss')

plt.legend(frameon=False)