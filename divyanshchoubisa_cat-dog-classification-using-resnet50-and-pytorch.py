import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

import warnings

warnings.simplefilter("ignore")

import torch 

import torch.nn as nn

from torch.utils.data import DataLoader



import torchvision



from torchvision import datasets, transforms

from torchvision.transforms import ToPILImage
from matplotlib.pyplot import imshow

import matplotlib.pylab as plt



from PIL import Image
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                 std=[0.229, 0.224, 0.225])



train_trans = transforms.Compose([

    transforms.RandomSizedCrop(224),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    normalize,

])



test_trans = transforms.Compose([

    transforms.Scale(256),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    normalize,

])
train_data = datasets.ImageFolder('../input/cat-and-dog/training_set/training_set/', transform=train_trans)

test_data = datasets.ImageFolder('../input/cat-and-dog/test_set/test_set/', transform=test_trans)
train_loader = torch.utils.data.DataLoader(train_data,

    batch_size=100,

    shuffle=True)

    



test_loader = torch.utils.data.DataLoader(test_data,

    batch_size=100,

    shuffle=False)
to_pil = transforms.ToPILImage()
def show_data(data_sample):

    result = to_pil(data_sample)

    plt.imshow(result)

    plt.title('y = '+ str(data_sample[1]))
for n,data_sample in enumerate(test_data):

    show_data(data_sample[0])

    plt.show()

    if n==2:

        break        
from torchvision.models import resnet50

model = resnet50(pretrained=True).cuda()

    

for param in model.parameters():

    param.requires_grad = False   

    

model.fc = nn.Sequential(

               nn.Linear(2048, 270),

               nn.ReLU(inplace=True),

               nn.Linear(270,90 ),

               nn.ReLU(inplace=True),

               nn.Linear(90,2)).cuda()
model
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.003)

optimizer
val_acc=[]

loss_list=[]

train_acc =[]

for epoch in range(10):

    loss_sublist = []

    corr = 0

    for x, y in train_loader:

        model.train()

        optimizer.zero_grad()

        x = x.cuda()

        y = y.cuda()

        preds = model(x)

        loss = criterion(preds, y)

        loss_sublist.append(loss.data.item())

        loss.backward()

        optimizer.step()

       

        _, yhat = torch.max(preds.data, 1)

        corr += (yhat == y).sum().item()

        print(".",end='',flush = True )

    

    loss_list.append(np.mean(loss_sublist))

    acc = corr/len(train_data)

    print("Training Accuracy at epoch",epoch+1,"is: ",acc)

    train_acc.append(acc)

    

    correct=0

    with torch.no_grad():

        for x_test, y_test in test_loader:

            model.eval()

            x_test = x_test.cuda()

            y_test = y_test.cuda()

            z = model(x_test)

            _, yhat1 = torch.max(z.data, 1)

            correct += (yhat1 == y_test).sum().item()

    accuracy = correct / len(test_data)

    print("Validation Accuracy at epoch",epoch+1,"is: ",accuracy)

    val_acc.append(accuracy)     
plt.figure(figsize=(14,7))

plt.plot(train_acc, label='Training Accuracy')

plt.plot(val_acc, label='Validation Accuracy')

plt.title('Accuracy Comparision')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()
plt.figure(figsize=(14,7))

plt.plot(loss_list,label='Training Loss')

plt.title('Training Loss Curve')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()
!mkdir models

!mkdir models/pytorch
torch.save(model.state_dict(), 'models/pytorch/model.pth')