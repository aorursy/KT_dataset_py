# importing Libraries

%matplotlib inline  

%config InlineBackend.figure_format='retina'

import torch #Importing Pytorch

import numpy as np  #importing Numerical Python Package

import matplotlib.pyplot as plt  #importing Mathmatical Plotting 
from torchvision import datasets ,transforms  #importing datasets and transforms from torchvision package

#datasets package is used to extract datasets fromt the web

#transform is used to do various transformations like cropping ,normaliasing,rotation etc.
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])

# creating the transform 1.creating pixels to tensor form 2.Normalising by image=(image-mean)/std i.e. 0.5 as mean and 0.5 as std
trainset=datasets.MNIST('~/.pytorch/MNIST_data/',train=True,transform=transform,download=True) #downloading the dataset and providing a path
trainloader=torch.utils.data.DataLoader(trainset,batch_size=64,shuffle=True)

# creating Trainloader that contain the batch of 64 images and images are shuffled to avoid biases
dataiter=iter(trainloader) # creating a iterator

images,label=dataiter.next() #creatng images and label variable i.e. image for image and label for image number i.e fom 0 to 9

print(images.shape) #finding shape as 64 batch 1 color channel and pixel size as 28*28

print(label.shape) # Label shape as target number i.e from 0 to 9
plt.imshow(images[0].numpy().squeeze(),cmap='Greys_r') #showing random image from images
from torch import nn #importing the neural net from torch
import torch.nn.functional as F # importing functional unit of Torch neural net
from torch import optim # importing the optimiser with default parameters
#Model creation with neural net Sequential model

model=nn.Sequential(nn.Linear(784,128), # 1 layer:- 784 input 128 o/p

                    nn.ReLU(),          # Defining Regular linear unit as activation

                    nn.Linear(128,64),  # 2 Layer:- 128 Input and 64 O/p

                    nn.ReLU(),          # Defining Regular linear unit as activation

                    nn.Linear(64,10),   # 3 Layer:- 64 Input and 10 O/P as (0-9)

                    nn.LogSoftmax(dim=1)) # Defining the log softmax to find the probablities for the last output unit
criterion=nn.NLLLoss()  #defining the negative log likelihood loss for calculating loss

optimizer=optim.SGD(model.parameters(),lr=0.005) # defining the optimiser with stochastic gradient descent and default parameters

epochs=5  # total number of iteration for training

for e in range(epochs):

    running_loss=0

    for images,labels in trainloader:

        images=images.view(images.shape[0],-1) # flatenning the images with size [64,784]

        optimizer.zero_grad()   #definin gradient in each epoch as 0

        output=model(images)     # modeling for each image batch

        loss=criterion(output,labels) #calculating the loss

        loss.backward()  # backpropagating the loss 

        optimizer.step()  #stepping the optimmizer again

        running_loss+=loss.item()  #calculating total loss in each epoch

    else:

        print(f"Training loss: {running_loss/len(trainloader)}") 

        

# shows the output as the total loss occur on each epoch and it can be seen loss decreases with each epoch    
# Showing result by taking a image as input and checking its result output

images, labels = next(iter(trainloader))



img = images[1].view(1, 784) # flatenning the image



with torch.no_grad():

    logps = model(img) #fitting in the model



ps = torch.exp(logps)  #calculating probability
plt.imshow(images[1].numpy().squeeze(),cmap='Greys_r') #showing the input image

plt.imshow(ps) #showing prediction as in form of bar graph