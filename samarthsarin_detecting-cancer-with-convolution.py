# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import torch

import torchvision

import torch.nn as nn

from torch.utils.data import DataLoader,Dataset

from torchvision.models import resnet50

import torchvision.transforms as transforms

from PIL import Image

import torch.optim as optim

from torch.utils.data.sampler import SubsetRandomSampler

import matplotlib.pyplot as plt

import cv2

%matplotlib inline
df = pd.read_csv('../input/histopathologic-cancer-detection/train_labels.csv')

print(df.shape)

df.head()
df['label'].value_counts()
class CancerDataset(Dataset):

    def __init__(self,csv,transform):

        self.data = pd.read_csv(csv)

        self.transform = transform

        self.label = torch.eye(2)[self.data['label']]

        

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self,idx):

        image_path = os.path.join('../input/histopathologic-cancer-detection/train/'+self.data.loc[idx]['id']+'.tif')

        image = Image.open(image_path)

        image = self.transform(image)

        label = torch.tensor(self.data.loc[idx]['label'])

        return {'images':image,'labels':label}
simple_transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([0.496,0.456,0.406],[0.229,0.224,0.225])])
train_dataset = CancerDataset('../input/histopathologic-cancer-detection/train_labels.csv',simple_transform)
data_size = len(train_dataset)

indices = list(range(data_size))

np.random.shuffle(indices)
split = int(np.round(0.2*data_size,0))
train_indices = indices[split:]

valid_indices = indices[:split]
train_sample = SubsetRandomSampler(train_indices)

valid_sample = SubsetRandomSampler(valid_indices)
train_loader = DataLoader(train_dataset,batch_size=32,sampler=train_sample)

valid_loader = DataLoader(train_dataset,batch_size=32,sampler=valid_sample)
model = resnet50(pretrained=False)

model.load_state_dict(torch.load('../input/resnet50/resnet50.pth'))

for param in model.parameters():

    param.require_grad = False

model.fc = nn.Linear(2048,2)

fc_parameters = model.fc.parameters()

for param in fc_parameters:

    param.require_grad = True



model = model.cuda()
criteria = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(),lr = 0.001)
def fit(epochs,model,optimizer,criteria):

    for epoch in range(epochs):

        training_loss = 0.0

        validation_loss = 0.0

        correct = 0

        total = 0

        

        print('{}/{} Epochs'.format(epoch+1,epochs))

        

        model.train()

        for batch_idx, d in enumerate(train_loader):

            data = d['images'].cuda()

            target = d['labels'].cuda()

            

            optimizer.zero_grad()

            output = model(data)

            loss = criteria(output,target)

            

            loss.backward()

            optimizer.step()

            

            training_loss = training_loss + ((1/(batch_idx+1))*(loss.data-training_loss))

            if batch_idx%1000 ==0:

                print('Training Loss is {}'.format(training_loss))

            pred = output.data.max(1,keepdim =True)[1]

            correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())

            total+= data.size(0)

            if batch_idx%1000 ==0:

                print('Batch ID of {} is having Training accuracy of {}'.format(batch_idx,(100*correct/total)))

         

        model.eval()

        for batch_idx,d in enumerate(valid_loader):

            data = d['images'].cuda()

            target = d['labels'].cuda()

            output = model(data)

            loss = criteria(output,target)

            validation_loss = validation_loss + ((1/(batch_idx+1))*(loss.data-validation_loss))

            if batch_idx%1000==0:

                print('Validation Loss is {}'.format(validation_loss))

            pred = output.data.max(1,keepdim=True)[1]

            correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())

            total+=data.size(0)

            if batch_idx%1000 ==0:

                print('Batch ID {} is having  validation accuracy of {}'.format(batch_idx,(100*correct/total)))

                

    return model
fit(5,model,optimizer,criteria)
class Prediction(Dataset):

    def __init__(self,csv,transform):

        self.data = pd.read_csv(csv)

        self.transform = transform

    

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self,idx):

        image_path = os.path.join('../input/histopathologic-cancer-detection/test/'+self.data.loc[idx]['id']+'.tif')

        image = Image.open(image_path)

        image = self.transform(image)

        return {'images':image}
test_dataset = Prediction('../input/histopathologic-cancer-detection/sample_submission.csv',simple_transform)
test_loader = DataLoader(test_dataset)
prediction_answer = []

for batch_idx,d in enumerate(test_loader):

    output = 0

    data = d['images'].cuda()

    output = model(data)

    output = output.cpu().detach().numpy()

    prediction_answer.append(np.argmax(output))
t = pd.read_csv('../input/histopathologic-cancer-detection/sample_submission.csv')
prediction_answer[:10]
len(prediction_answer)
len(prediction_answer)
t['label'] = prediction_answer

t.head(20)
t.to_csv('submission.csv',index = False)