import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

print(os.listdir("../input"))
import torch

from torchvision import models, datasets, transforms

import torch.nn as nn

import torch.functional as F



from torch.utils.data import Dataset, DataLoader

from PIL import Image

import matplotlib.pyplot as plt



anno_train=pd.read_csv("../input/anno_train.csv")

anno_train.head(5)
train_directory="../input/car_data/car_data/train"

test_directory="../input/car_data/car_data/test"
train_transform=transforms.Compose([

    transforms.RandomRotation(30),

    transforms.RandomResizedCrop(224),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize((0.485, 0.456,0.406),(0.229,0.225,0.224))

])



test_transform=transforms.Compose([

    transforms.Resize(255),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize((0.485,0.456, 0.406 ), (0.229, 0.225,0.224))

])
trainset=datasets.ImageFolder(train_directory,transform=train_transform)

testset=datasets.ImageFolder(test_directory,transform=test_transform)
trainloader=DataLoader(trainset, shuffle=True, batch_size=64)

testloader=DataLoader(testset, shuffle=True, batch_size=64)



train_itr=iter(trainloader)

test_itr=iter(testloader)
train_sample_ftrs, train_sample_lbls=train_itr.next()

print(train_sample_ftrs.shape)

#print(train_sample_ftrs[0])



print(train_sample_lbls)

print(train_sample_lbls.max())

print(train_sample_lbls.min())
# A function that will be used to view our images

def plot_image(image_tensor):

    plt.figure()

    plt.imshow(image_tensor.numpy().transpose(1,2,0))

    plt.show()
print(train_sample_lbls.min())
%matplotlib inline

tensor_image=train_sample_ftrs[0].view(3,224,224)

plot_image(tensor_image)
model=models.resnet101(pretrained=True)

for param in model.parameters():

    param.requires_grad=False

    

classifier=nn.Sequential(nn.Linear(2048, 700),

                          nn.ReLU(),

                          nn.Dropout(p = 0.2),

                          nn.Linear(700 , 300),

                          nn.ReLU(),

                          nn.Dropout(p = 0.2),

                          nn.Linear(300 ,196),

                          nn.LogSoftmax(dim=1))

model.fc=classifier

    

print(model.fc)

model=model.cuda()

#model = nn.DataParallel(model)



    
from torch import optim



criterion=nn.NLLLoss()

optimizer=optim.Adam(model.fc.parameters(),lr=0.003)
def train(no_epochs):

  for e in range(no_epochs):

    model.train()

    running_loss=0

    valid_loss=0    

    for images, labels in trainloader:

        optimizer.zero_grad()

        #print(images.size())

        images=images.cuda()

        labels=labels.cuda()

        #print(output.size)

        #print(labels.size)



        output=model.forward(images)

        loss=criterion(output,labels)

        loss.backward()

        

        running_loss+=loss.item()

        optimizer.step()

        

    else:

        model.eval()

        accuracy=0

        for images,labels in testloader:

            images=images.cuda()

            labels=labels.cuda()

            

            output=model(images)

            loss=criterion(output, labels)

            loss.backward()

            valid_loss+=loss.item()

            

            log_ps=torch.exp(output)

            top_p,top_class=log_ps.topk(1,dim=1)

            equality=top_class==labels.view(*top_class.shape)

            accuracy+=torch.mean(equality.type(torch.FloatTensor))

    

    print("epoch "+str(e+1)+" : training loss: "+str(running_loss/len(trainloader))+" testing loss: "+str(valid_loss/len(validloader))+" Accuracy: "+str(accuracy/len(validloader))) 

        

    
train(2)