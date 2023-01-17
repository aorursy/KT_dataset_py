import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import cv2

import torch

import torch.nn.functional as F

import torch.nn as nn

from torchvision import transforms,datasets
#Check if GPU is present or not and print its name

torch.cuda.get_device_name()
train  = datasets.MNIST("",train=True,download=True,transform= transforms.Compose([transforms.ToTensor()]))

test  = datasets.MNIST("",train=False,download=True,transform= transforms.Compose([transforms.ToTensor()]))
trainset = torch.utils.data.DataLoader(train,batch_size=32,shuffle=True)

testset = torch.utils.data.DataLoader(test,batch_size=32,shuffle=True)
for data in trainset:

    print(data)

    break
x,y = data[0][0],data[0][1]
%matplotlib inline 

plt.imshow(x.view(28,28),alpha = 0.8,cmap='gist_gray')

plt.show()
#Create an empty dictionary to hold count of each digit.

total = 0

counter_dict = {}

for i in range(10):

  counter_dict[i]=0

print(counter_dict)
#Store the count of each digit stored . This step help us to check whether data is imbalanced or not.

for data in trainset:

  Xs,ys = data

  for y in ys:

    counter_dict[int(y)]+=1

print(counter_dict)
#Create a Basic Dense layer Model

class Net(nn.Module):

  def __init__(self):

    super().__init__() #Intialising the parent class(basically inheritance)

    self.fc1 = nn.Linear(28*28,64) #Dense Layers

    self.fc2 = nn.Linear(64,64)

    self.fc3 = nn.Linear(64,64)

    self.fc4 = nn.Linear(64,10) 



  def forward(self,x):

    x = F.relu(self.fc1(x))

    x = F.relu(self.fc2(x))

    x = F.relu(self.fc3(x))

    x = self.fc4(x)

    return F.log_softmax(x,dim=1) #Final Activation Function (gives probablity of each class)



net = Net()

print(net)
X = torch.rand([28,28]) #Apply Random input to the model

X
output = net(X.view(-1,28*28)) #Reshape the output (same as numpy reshape function)

output
import torch.optim as optim

from tqdm import tqdm

optimizer = optim.Adam(net.parameters(),lr=0.001) #Optimizer Function to update the weight for forward function



epochs = 10

for epoch in range(epochs):

  for data in tqdm(trainset):

    X,y = data

    net.zero_grad()

    output = net(X.view(-1,28*28))

    loss = F.nll_loss(output,y)

    loss.backward()

    optimizer.step()

  print("Epoch: {}, Loss: {}".format(epoch,loss))
correct = 0

total = 0



with torch.no_grad():

  for data in trainset:

    X,y = data

    output = net(X.view(-1,28*28))

    for idx,i in enumerate(output):

      if torch.argmax(i)==y[idx]:

        correct+=1

      total+=1

print("Accuracy: {}".format(correct/total))
fig = plt.figure(figsize=(10,10))

for i in range(1,10):

  fig.add_subplot(5,5,i)

  plt.imshow(X[i].view(28,28),alpha= 0.8,cmap='gist_gray')

  print(torch.argmax(net(X[i].view(-1,784))[0]))

plt.show()