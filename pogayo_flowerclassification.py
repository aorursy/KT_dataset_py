

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
print(os.listdir("../input/flowerclassification"))

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, utils, datasets, models

from PIL import Image

import matplotlib.pyplot as plt

cpt=torch.load("../input/flowerclassification/checkpoint4.pth")
train_transforms = transforms.Compose([transforms.RandomRotation(30),

                                      transforms.RandomResizedCrop(224),

                                      transforms.RandomHorizontalFlip(),

                                      transforms.ToTensor(),

                                      transforms.Normalize((0.485,0.456,0.406),(0.229,0.225,0.224))])

valid_transforms = transforms.Compose([transforms.Resize(255),

                                      transforms.CenterCrop(224), 

                                      transforms.ToTensor(), 

                                      transforms.Normalize((0.485,0.456,0.406),(0.229,0.225,0.224))])



train_directory="../input/hackathon-blossom-flower-classification/flower_data/flower_data/train"

valid_directory="../input/hackathon-blossom-flower-classification/flower_data/flower_data/valid"



trainset=datasets.ImageFolder(train_directory, transform=train_transforms)

validset=datasets.ImageFolder(valid_directory, transform=valid_transforms)

trainloader=DataLoader(trainset, shuffle=True, batch_size=64)

validloader=DataLoader(validset, shuffle=True, batch_size=64)
train_itr=iter(trainloader)

sample, sample_label=train_itr.next()

print(sample.shape)

print(sample_label.shape)

print(torch.max(sample_label))



def plot_image(tensor):

    plt.figure()

    plt.imshow(tensor.numpy().transpose(1,2,0))

    plt.show
tensor_image=sample[12].view(3,224,224)

to_pil = transforms.ToPILImage()

img = to_pil(tensor_image)

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

                          nn.Linear(300 ,102),

                          nn.LogSoftmax(dim=1))

model.fc=classifier

    

print(model.fc)

model.cuda()
model.load_state_dict(cpt)
from torch import optim

criterion=nn.NLLLoss()

optimizer=optim.Adam(model.fc.parameters(), lr=0.007)

itr=iter(validloader)

sample_test, sample_label=itr.next()

#print(sample_test[0])

print(torch.max(sample_label))

print(torch.cuda.is_available())
def train(no_epochs):

  for e in range(no_epochs):

    model.train()

    running_loss=0

    valid_loss=0    

    for images, labels in trainloader:

        optimizer.zero_grad()

        images=images.cuda()

        labels=labels.cuda()



        output=model.forward(images)

        loss=criterion(output,labels)

        loss.backward()

        

        running_loss+=loss.item()

        optimizer.step()

        

    else:

        model.eval()

        accuracy=0

        for images,labels in validloader:

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

        

    
train(1)
torch.save(model.state_dict(),"checkpoint.pth")