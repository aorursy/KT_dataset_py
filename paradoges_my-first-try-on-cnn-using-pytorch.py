

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision

import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.
#Reading MNIST Datasets
path="../input/"
trainset=pd.read_csv(path+'train.csv')
testset=pd.read_csv(path+'test.csv')
#Defining hyperparameters
batchsize=200 #size of minibatch
Epoch=100 #the number iteration
LR=0.001 #learning rate
#Data preprocessing
trainset_x=np.array(trainset.iloc[:,1:]) #Turn trainset features series into array
trainset_y=np.array(trainset['label']) #Turn trainset labels serie into array

trainset_x=np.reshape(trainset_x,(-1,batchsize,28,28)) #Reshape trainset features into 210*200*28*28
trainset_y=trainset_y.reshape(-1,batchsize) #Reshape trainset labels into 210*200

trx=torch.from_numpy(trainset_x) #Turn array into tensor
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.ksize_conv=5 #kernel size of Convolution Layers
        self.ksize_maxpool=2 #kernel size of Max Pool Layers
        self.stride_conv=1 #the step of Convolution Kernel
        self.padding=2 #padding number of Convolution Layers
        self.channel_conv1=16 #output channel number of Convolution Layer 1
        self.channel_conv2=32 #output channel number of Convolution Layer 1
        # Convolution layer 1
        self.conv1=nn.Sequential(nn.Conv2d(1,self.channel_conv1,self.ksize_conv,stride=self.stride_conv,padding=self.padding),#input shape(1,28,28),output shape(16,28,28)
                                 nn.BatchNorm2d(self.channel_conv1), #Batch normalization
                                nn.ReLU(), #Activation layer
                                nn.MaxPool2d(self.ksize_maxpool)) #input shape(16,28,28),output shape(16,14,14)
        self.conv2=nn.Sequential(nn.Conv2d(self.channel_conv1,self.channel_conv2,self.ksize_conv,stride=self.stride_conv,padding=self.padding),#input shape(16,14,14),output shape(32,14,14)
                                 nn.BatchNorm2d(self.channel_conv2), #Batch normalization
                                nn.ReLU(), #Activation layer
                                nn.MaxPool2d(self.ksize_maxpool)) #input shape(32,14,14),output shape(32,7,7)
        self.out=nn.Linear(32*7*7,10) #fully connected layer, output 10 classes
        
    def forward(self,x): #forward propagation
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(x.size(0),-1) #flattern tensor into (batchsize,32*7*7)
        output = self.out(x)
        return output
    
cnn=CNN() #bulid CNN model    
optimizer = torch.optim.Adam(cnn.parameters(),lr=LR) # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss() #softmax and cross entropy loss
for epoch in range(Epoch):
    for step in range(0,42000//batchsize):
        x=Variable(trx[step].view(batchsize,1,28,28).float()) #warp tarinning set feature into Variable
        y=Variable(torch.Tensor(trainset_y[step]).long()) #warp tarinning set label into Variable
        output = cnn(x) #fit the model
        loss = loss_func(output,y) #calculate the loss
        optimizer.zero_grad() # clear gradients for this training step
        loss.backward() #backward propagation
        optimizer.step() # apply gradients
        #print('Epoch: ', epoch, 'Batch: ', step, '| train loss: %.4f' % loss.data.item())
    print('Epoch: ', epoch, 'Batch: ', step, '| train loss: %.4f' % loss.data.item())

test_x=Variable(torch.Tensor(np.array(testset)).view(-1,1,28,28))
output=cnn(test_x)
y = torch.max(output,1)[1].data.numpy().squeeze()
y
submit=pd.read_csv(path+'sample_submission.csv')
submit['Label']=y
submit
submit.to_csv(path_or_buf='submission.csv',index=False)