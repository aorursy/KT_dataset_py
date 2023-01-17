import numpy as np

import torch

import helper

from PIL import Image

import pandas as pd

import numpy as np

import os

import keras

import matplotlib.pyplot as plt

import cv2

import warnings

warnings.filterwarnings("ignore")

import math
img1 = cv2.imread("../input/espectograma/Train/Annette.0.jpg")

img2 = cv2.imread("../input/espectograma/Train/Sebas.0.jpg")

img3 = cv2.imread("../input/espectograma/Train/Vale.0.jpg")



imgs = np.array([[img1], [img2], [img3]])



train = torch.from_numpy(imgs)



print(imgs.shape)



def activation(x):

    return 1/(1 + torch.exp(-x))



inputs = train.view(train.shape[0], -1)

inputs = inputs.float()



w1 = torch.randn(307200, 320) #cambie el tamaño para multiplicar inputs*w1

b1 = torch.randn(320)

#w1 = w1.byte()

#print (w1.size())

#print (inputs.size())



#w2 = torch.randn(3200, 3)

#b2 = torch.randn(3)



w2 = torch.randn(320, 3)  #prueba

b2 = torch.randn(3)



h = activation(torch.mm(inputs, w1) + b1)

out = torch.mm(h, w2) + b2

print(out)
def softmax(x):

    return torch.exp(x)/torch.sum(torch.exp(x), dim=1).view(-1,1)



probabilities = softmax(out)

print(probabilities.shape)

print(probabilities.sum(dim=1))

print(probabilities)
from torch import nn


class Network(nn.Module):

    def __init__(self):

        super().__init__()

        

        # Inputs to hidden layer linear transformation

        self.hidden = nn.Linear(784, 256)

        # Output layer, 10 units - one for each digit

        self.output = nn.Linear(256, 10)

        

        # Define sigmoid activation and softmax output 

        self.sigmoid = nn.Sigmoid()

        self.softmax = nn.Softmax(dim=1)

        

    def forward(self, x):

        # Pass the input tensor through each of our operations

        x = self.hidden(x)

        x = self.sigmoid(x)

        x = self.output(x)

        x = self.softmax(x)

        

        return x
model = Network()

model


import torch.nn.functional as F



class Network(nn.Module):

    def __init__(self):

        super().__init__()

        # Inputs to hidden layer linear transformation

        self.hidden = nn.Linear(784, 256)

        # Output layer, 10 units - one for each digit

        self.output = nn.Linear(256, 10)

        

    def forward(self, x):

        # Hidden layer with sigmoid activation

        x = F.sigmoid(self.hidden(x))

        # Output layer with softmax activation

        x = F.softmax(self.output(x), dim=1)

        

        return x
class Network(nn.Module):

    def __init__(self):

        super().__init__()

        # Defining the layers, 128, 64, 10 units each

        self.fc1 = nn.Linear(784, 128)

        self.fc2 = nn.Linear(128, 64)

        # Output layer, 10 units - one for each digit

        self.fc3 = nn.Linear(64, 10)

        

    def forward(self, x):

        ''' Forward pass through the network, returns the output logits '''

        

        x = self.fc1(x)

        x = F.relu(x)

        x = self.fc2(x)

        x = F.relu(x)

        x = self.fc3(x)

        x = F.softmax(x, dim=1)

        

        return x



model = Network()

model
print(model.fc1.weight)

print(model.fc1.bias)


# Set biases to all zeros

model.fc1.bias.data.fill_(0)
# sample from random normal with standard dev = 0.01

model.fc1.weight.data.normal_(std=0.01)


# Grab some data 

dataiter = iter(imgs)

images = next(dataiter)



# Resize images into a 1D vector, new shape is (batch size, color channels, image pixels) 

imgs.resize((64, 1, 784), refcheck = False)

# or images.resize_(images.shape[0], 1, 784) to automatically get batch size



# Forward pass through the network

img_idx = 0

#ps = model.forward(imgs[img_idx,:])



img = imgs[img_idx]

helper.view_classify(img.view(1, 320, 320), ps)