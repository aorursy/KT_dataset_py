import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



#Pytorch 

import torch

import torchvision

import torch.nn.functional as F

from torch import nn, optim

from torch.optim.lr_scheduler import StepLR

from torch.utils.data import Dataset, DataLoader

from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import transforms, models



from PIL import Image

from os import listdir

from os.path import isfile, join

import cv2

from sklearn.model_selection import train_test_split



import warnings

warnings.filterwarnings('ignore')



%matplotlib inline



import os

print(os.listdir("../input"))

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('device: {}'.format(device))

train_folder='../input/train/train/'

test_folder='../input/test//test/'

labels=pd.read_csv('../input/train.csv')

submission=pd.read_csv('../input/sample_submission.csv')

print('train dataset size: {}'.format(len(listdir(train_folder))))

print('test dataset size: {}'.format(len(listdir(test_folder))))
#Create simple dataset

class Cactus(Dataset):

    def __init__(self, folder, labels, transform=None):

        self.transform=transform

        self.folder=folder

        self.labels=labels

    def __len__(self):

        return self.labels.shape[0]

    def __getitem__(self,index):

        img_path=os.path.join(self.folder, self.labels['id'].iloc[index])

        img=Image.open(img_path)

        img_label=self.labels['has_cactus'].iloc[index]

        if self.transform:

            img=self.transform(img)

        return img, img_label
# Visualize some samples and their transformations

def visualize_samples(dataset, indices, count=5):

    plt.figure(figsize=(count*3,3))

    display_indices = indices[:count]

    for i, index in enumerate(display_indices):    

        x, y = dataset[index]

        plt.subplot(1,count,i+1)

        plt.title('has cactus: {}'.format(y))

        plt.imshow(x)

        plt.grid(False)

        plt.axis('off')   



orig_dataset=Cactus(train_folder, labels)

indices = np.random.choice(np.arange(len(orig_dataset)), 5, replace=False)

visualize_samples(orig_dataset, indices)
transformed_dataset=Cactus(train_folder, labels, 

                                    transform=transforms.Compose([

                                    transforms.RandomHorizontalFlip(),

                                    transforms.RandomVerticalFlip(),

                                    transforms.RandomAffine(10),

                                    transforms.RandomRotation((-10,10))

                                                                    ]))

visualize_samples(transformed_dataset, indices)
#Transforms

tfs_train=transforms.Compose([

    transforms.RandomHorizontalFlip(),

    transforms.RandomVerticalFlip(),

    transforms.RandomRotation((-10,10)),

    transforms.ToTensor(),

    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))                

                      ])

tfs_test=transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))                

                      ])



#Datasets

train_dataset=Cactus(train_folder, labels, transform=tfs_train)

test_dataset=Cactus(test_folder, submission, transform=tfs_test)



# Dividing train dataset into train and validation

bs=128

data_size=len(train_dataset)

split=int(np.floor(0.1*data_size))

data_indices=list(range(data_size))

np.random.shuffle(data_indices)

train_indices, val_indices=data_indices[split:], data_indices[:split]



#Samplers

train_sampler=SubsetRandomSampler(train_indices)

val_sampler=SubsetRandomSampler(val_indices)



#Loaders

train_loader=DataLoader(train_dataset, batch_size=bs, sampler=train_sampler)

val_loader=DataLoader(train_dataset, batch_size=bs//2, sampler=val_sampler)

test_loader=DataLoader(test_dataset, batch_size=bs//2)
def train_model(model, train_loader, val_loader, loss, optimizer, num_epoch=10):

    print('Training...')

    loss_history=[]

    train_history=[]

    val_history=[]

    val_loss_history=[]

    for epoch in range(num_epoch):

        model.train()

        loss_accum=0

        correct_samples=0

        total_samples=0

        

        for i_step, (x,y) in enumerate(train_loader):

            x,y=x.to(device),y.to(device)

            prediction=model(x)

            loss_value=criterion(prediction,y)

            optimizer.zero_grad()

            loss_value.backward()

            optimizer.step()

            

            _, indices=torch.max(prediction, 1)

            correct_samples+=torch.sum(indices==y).item()

            total_samples+=y.shape[0]

            

            loss_accum+=loss_value

            

        scheduler.step()                    

        train_loss=loss_accum/i_step

        train_accuracy=float(correct_samples)/total_samples

        val_accuracy, val_loss=compute_accuracy(model, val_loader)

        

        loss_history.append(float(train_loss))

        train_history.append(train_accuracy)

        val_history.append(val_accuracy)

        val_loss_history.append(val_loss)

        

        print('Epoch: {}/{}, Train_loss: {}, Train_accuracy: {}, Val_loss: {}, Val_accuracy: {}'.format(epoch+1,

                            num_epoch,train_loss,train_accuracy,val_loss,val_accuracy))

    return loss_history, train_history, val_history, val_loss_history



def compute_accuracy(model,loader):

    model.eval()

    correct_samples=0

    total_samples=0

    loss_accum=0

    with torch.no_grad():

        for i_step, (x,y) in enumerate(loader):

            x,y=x.to(device),y.to(device)

            prediction=model(x)

            loss_value=criterion(prediction,y)

            _,indices=torch.max(prediction,1)

            correct_samples+=torch.sum(indices==y).item()

            total_samples+=y.shape[0]

            loss_accum+=loss_value

    ave_loss=loss_accum/i_step

    accuracy=float(correct_samples)/total_samples

    return accuracy, ave_loss
#Define model

model=models.resnet18(pretrained=True)

num_ftrs=model.fc.in_features

model.fc=nn.Linear(num_ftrs,2)

model=model.to(device)



#Start training

criterion=nn.CrossEntropyLoss()

optimizer=optim.Adamax(model.parameters(), lr=0.003, weight_decay=0.001)

scheduler=StepLR(optimizer, step_size=10, gamma=0.1)

loss_history, train_history, val_history, val_loss_history = train_model(model, train_loader, 

                                        val_loader, criterion, optimizer, 30)
#Create submission file

model.eval()



predictions=[]

for i, (x,y) in enumerate(test_loader):

    x,y=x.to(device), y.to(device)

    prediction=model(x)

    pred=prediction[:,1].detach().cpu().numpy()

    for i in pred:

        predictions.append(i)



submission['has_cactus']=predictions

submission.to_csv('submission.csv',index=False)