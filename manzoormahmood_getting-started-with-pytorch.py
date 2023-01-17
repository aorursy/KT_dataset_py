import cv2

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import os

import math

%matplotlib inline

import time



#pytorch utility imports

import torch

import torchvision

import torchvision.transforms as transforms

from torch.utils.data import DataLoader, TensorDataset

from torchvision.utils import make_grid



#neural net imports

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.autograd import Variable



start = torch.cuda.Event(enable_timing=True) #time measure during cuda training

end = torch.cuda.Event(enable_timing=True)
test_df = pd.read_csv('../input/mnist-in-csv/mnist_test.csv')

train_df = pd.read_csv('../input/mnist-in-csv/mnist_train.csv')

train_df.head()
train_labels = train_df['label'].values # converting to numpy also



test_labels=test_df['label'].values

train_images = (train_df.iloc[:,1:].values).astype('float32')

test_images = (test_df.iloc[:,1:].values).astype('float32')
print("train images shape",train_images.shape)

print("train labels shape",train_labels.shape)

print("test images shape",test_images.shape)

print("test labels shape",test_labels.shape)
train_images = train_images.reshape(train_images.shape[0], 28, 28)

test_images = test_images.reshape(test_images.shape[0], 28, 28)

print(train_images.shape)

print(test_images.shape)
#train samples

for i in range(6, 9):

    plt.subplot(330 + (i+1))

    plt.imshow(train_images[i].squeeze(), cmap=plt.get_cmap('gray'))

    plt.title(train_labels[i])

train_images_tensor = torch.tensor(train_images)/255.0 #default torch.FloatTensor

train_labels_tensor = torch.tensor(train_labels)

train_tensor = TensorDataset(train_images_tensor, train_labels_tensor)



test_images_tensor = torch.tensor(test_images)/255.0

test_labels_tensor = torch.tensor(test_labels)

test_tensor = TensorDataset(test_images_tensor, test_labels_tensor)
train_loader = DataLoader(train_tensor, batch_size=16, num_workers=2, shuffle=True)

test_loader = DataLoader(test_images_tensor, batch_size=16, num_workers=2, shuffle=False)
class Model(nn.Module):

    def __init__(self):

        super(Model, self).__init__()

        

        self.fc1 = nn.Linear(784, 548)

        self.bc1 = nn.BatchNorm1d(548)

        

        self.fc2 = nn.Linear(548, 252)

        self.bc2 = nn.BatchNorm1d(252)

        

        self.fc3 = nn.Linear(252, 10)

        

        

    def forward(self, x):

        x = x.view((-1, 784))

        h = self.fc1(x)

        h = self.bc1(h)

        h = F.relu(h)

        h = F.dropout(h, p=0.5, training=self.training)

        

        h = self.fc2(h)

        h = self.bc2(h)

        h = F.relu(h)

        h = F.dropout(h, p=0.2, training=self.training)

        

        h = self.fc3(h)

        out = F.log_softmax(h)

        return out



model = Model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)



if (device.type=='cuda'):

    model.cuda() # convert model to cuda model



    

optimizer = optim.Adam(model.parameters(), lr=0.001) #adam optimizer from optim module
if (device.type=='cuda'):

    start.record() #timer start



model.train()





losses = []

for epoch in range(20):

    for batch_idx, (data, target) in enumerate(train_loader):

        # Get Samples

        if (device.type=='cuda'):

            data, target = Variable(data.cuda()), Variable(target.cuda())

        else:

            data, target = Variable(data), Variable(target) # making group of 16

            

        

        # Init

        optimizer.zero_grad() #making gradient zero for new mini-batch. 



        # Predict

        y_pred = model(data) 

         

        

        # Calculate loss

        loss = F.cross_entropy(y_pred, target)

        losses.append(loss.data)

        

        # Backpropagation

        loss.backward()  #It computes gradient of loss w.r.t all the parameters and store them in (parameter.grad) attribute.

        optimizer.step() #optimizer.step() updates all the parameters based on (parameter.grad)

        

        

        # Display

        #if batch_idx % 100 == 1:

        print('\r Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format( epoch, batch_idx * len(data), len(train_loader.dataset),

                100. * batch_idx / len(train_loader), loss.data), end='')

            

    print()

    

if (device.type=='cuda'):

    end.record()
if (device.type=='cuda'):

    evaluate_x=test_images_tensor.cuda()

    evaluate_y=test_labels_tensor.cuda()

else:

    evaluate_x=test_images_tensor

    evaluate_y=test_labels_tensor

    



output = model(evaluate_x)



pred = output.data.max(1)[1]

d = pred.eq(evaluate_y.data).cpu()

a=(d.sum().data.cpu().numpy())

b=d.size()

b=torch.tensor(b)

b=(b.sum().data.cpu().numpy())

accuracy = a/b



print('Accuracy:', accuracy)
if (device.type=='cuda'):

    torch.cuda.synchronize()

    print(start.elapsed_time(end)/1000,"sec")
test_df = pd.read_csv('../input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv')

train_df = pd.read_csv('../input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv')
print((train_df['label'].unique()).shape )# There are 24 possible labels, 9=J and 25=Z require motion so they are absent.

print(np.sort(train_df['label'].unique()))
train_labels = train_df['label'].values

test_labels=test_df['label'].values

train_images = (train_df.iloc[:,1:].values).astype('float32')

test_images = (test_df.iloc[:,1:].values).astype('float32')
print("train images shape",train_images.shape)

print("train labels shape",train_labels.shape)

print("test images shape",test_images.shape)

print("test labels shape",test_labels.shape)
train_images = train_images.reshape(train_images.shape[0],1, 28, 28)

test_images = test_images.reshape(test_images.shape[0],1, 28, 28)
print(train_images.shape)

print(test_images.shape)
train_images_tensor = torch.tensor(train_images)/255.0 #default torch.FloatTensor

train_labels_tensor = torch.tensor(train_labels)

train_tensor = TensorDataset(train_images_tensor, train_labels_tensor)



test_images_tensor = torch.tensor(test_images)/255.0

test_labels_tensor = torch.tensor(test_labels)

test_tensor = TensorDataset(test_images_tensor, test_labels_tensor)
train_loader = DataLoader(train_tensor, batch_size=16, num_workers=2, shuffle=True)

test_loader = DataLoader(test_images_tensor, batch_size=16, num_workers=2, shuffle=False)
import torch.nn.functional as F

from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout

from torch.optim import Adam, SGD



class Net(nn.Module):                                           # class Net inherits from predefined Module class in torch.nn

    def __init__(self):                                         # calling constructor of  parent class

        super().__init__()                                     

        

        

        self.conv1 = nn.Conv2d(1,32,3)              # 2d convolution layer : (input : 1 image , output : 32 channels , kernel size : 3*3)

        self.conv2 = nn.Conv2d(32,64,3)

        self.conv3 = nn.Conv2d(64,128,3)

        

        self.linear_in = None                      # used to calculate input of first linear layer by passing fake data through 2d layers

        x = torch.rand(28,28).view(-1,1,28,28)     # using convs function

        self.convs(x)

    

        self.fc1 = nn.Linear(self.linear_in,512)

        self.fc2 = nn.Linear(512,26)

        

    def convs(self,x):

        x = F.max_pool2d(F.relu(self.conv1(x)) , (2,2) )      # relu used for activation function 

        x = F.max_pool2d(F.relu(self.conv2(x)) , (2,2) )      # max_pool2d for max pooling results of each kernel with window size 2*2

        x = F.max_pool2d(F.relu(self.conv3(x)) , (2,2) )

        

        if self.linear_in == None:

            self.linear_in = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]  # input of first linear layer is multiplication of dimensions of ouput 

        return x                                                        # tensor of the 2d layers

    

    def forward(self,x):                                    # forward pass function uses the convs function to pass through 2d layers

        x = self.convs(x)

        x = x.view(-1,self.linear_in)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        x = F.log_softmax(x ,dim = -1)                     # log_softmax for finding output neuron with highest value

        return x

    

net = Net()
print(net)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)



if (device.type=='cuda'):

    model.cuda() # CUDA



net.to(device)





import torch.optim as optim



criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=0.001)
if (device.type=='cuda'):

    start.record() 

    

loss_log = []

for epoch in range(20): # loop over dataset multiple times

    running_loss = 0.0

    for i, (data,target) in enumerate(train_loader):



        

        if (device.type=='cuda'):

            inputs,labels= Variable(data.cuda()), Variable(target.cuda())

        else:

            inputs,labels= Variable(data), Variable(target)

       

        

        

        # zero parameter gradients

        optimizer.zero_grad()

        

        # forward + backward + optimize

        outputs = net(inputs)



        loss =  F.cross_entropy(outputs, labels)

        #print(loss)

        

   

        

        loss.backward()

        optimizer.step()

        

      

        #if i % 100 == 1:

        print('\r Train Epoch: {} [{}/{} ({:.0f}%)] \tLoss: {:.6f}'.format( epoch, i * len(data), len(train_loader.dataset),

                                                                           100. * i / len(train_loader), loss.data), end='')

        

    print("")

                

print('Finished Training')

if (device.type=='cuda'):

    end.record()
if (device.type=='cuda'):

    evaluate_x=test_images_tensor.cuda()

    evaluate_y=test_labels_tensor.cuda()

else:

    evaluate_x=test_images_tensor

    evaluate_y=test_labels_tensor

    



output = net(evaluate_x)



pred = output.data.max(1)[1]

d = pred.eq(evaluate_y.data).cpu()

a=(d.sum().data.cpu().numpy())

b=d.size()

b=torch.tensor(b)

b=(b.sum().data.cpu().numpy())

accuracy = a/b



print('Accuracy:', accuracy*100)
if (device.type=='cuda'):

    torch.cuda.synchronize()

    print(start.elapsed_time(end)/1000,"sec")
from sklearn.metrics import f1_score

print("f1 score =",f1_score(test_labels, pred.cpu().numpy(), average='macro'))