import torchvision

import  torch.nn as nn

import torch

import torch.nn.functional as F

from torchvision import transforms,models,datasets

import matplotlib.pyplot as plt

from PIL import Image

import numpy as np

from torch import optim
train_data_dir = '/kaggle/input/cat-and-dog/training_set/training_set'



transform = transforms.Compose([transforms.Resize(255),

                                transforms.CenterCrop(224),

                                transforms.ToTensor()])



dataset = torchvision.datasets.ImageFolder(train_data_dir, transform= transform)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=400 ,shuffle=True)
test_data_dir = '/kaggle/input/cat-and-dog/test_set/test_set'



transform = transforms.Compose([transforms.Resize(255),

                                transforms.CenterCrop(224),

                                transforms.ToTensor()])



dataset = torchvision.datasets.ImageFolder(train_data_dir, transform= transform)

test_loader = torch.utils.data.DataLoader(dataset, batch_size=400 ,shuffle=True)
def imshow(inp, title=None):

    """Imshow for Tensor."""

    inp = inp.numpy().transpose((1, 2, 0))

    plt.figure(figsize=(20,150))

    plt.imshow(inp)



inputs, classes = next(iter(train_loader))



# Make a grid from batch

out = torchvision.utils.make_grid(inputs, scale_each= True)



imshow(out)
model = models.densenet121(pretrained = True)
for params in model.parameters():

    params.requires_grad = False
from collections import OrderedDict



classifier = nn.Sequential(OrderedDict([

    ('fc1',nn.Linear(1024,500)),

    ('relu',nn.ReLU()),

    ('fc2',nn.Linear(500,2)),

    ('Output',nn.LogSoftmax(dim=1))

]))



model.classifier = classifier
model = model.cuda()
optimizer= optim.Adam(model.classifier.parameters())

criterian= nn.NLLLoss()

list_train_loss=[]

list_test_loss=[]



for epoch in range(10):

    train_loss= 0

    test_loss= 0

    for bat,(img,label) in enumerate(train_loader):

        

        # moving batch and lables to gpu

        img = img.to('cuda:0')

        label = label.to('cuda:0')

        

        model.train()

        optimizer.zero_grad()



        output = model(img)

        loss = criterian(output,label)

        loss.backward()

        optimizer.step()

        train_loss = train_loss+loss.item()

        #print(bat)



    accuracy=0

    for bat,(img,label) in enumerate(test_loader):

        img = img.to('cuda:0')

        label = label.to('cuda:0')



        model.eval()

        logps= model(img)

        loss = criterian(logps,label)



        test_loss+= loss.item()

        ps=torch.exp(logps)

        top_ps,top_class=ps.topk(1,dim=1)

        equality=top_class == label.view(*top_class.shape)

        accuracy +=torch.mean(equality.type(torch.FloatTensor)).item()



    list_train_loss.append(train_loss/20)

    list_test_loss.append(test_loss/20)

    print('epoch: ',epoch,'    train_loss:  ',train_loss/20,'   test_loss:    ',test_loss/20,'    accuracy:  ',accuracy/len(test_loader))