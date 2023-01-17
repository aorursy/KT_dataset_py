# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import torch

import torchvision

import torch.nn as nn

from torchvision.models import resnet101

from torch.utils.data import DataLoader,Dataset

from torch.utils.data.sampler import SubsetRandomSampler

import torchvision.transforms as transforms

from PIL import Image

import torch.optim as optim

from torch.optim import lr_scheduler
df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

print(df.shape)

df.head()
class EyeDataset(Dataset):

    def __init__(self,csv,transform):

        self.data = pd.read_csv(csv)

        self.transform = transform

        self.labels = torch.eye(5)[self.data['diagnosis']]

        

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self,idx):

        image_path = os.path.join('../input/aptos2019-blindness-detection/train_images/'+self.data.loc[idx]['id_code']+'.png')

        image = Image.open(image_path)

        image = self.transform(image)

        label = torch.tensor(self.data.loc[idx]['diagnosis'])

        return {'images':image,'labels':label}
data_transform = transforms.Compose([ transforms.RandomResizedCrop((224,224)),

                                     transforms.RandomHorizontalFlip(),

                                     transforms.ToTensor(),

                                     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
train_dataset = EyeDataset('../input/aptos2019-blindness-detection/train.csv',data_transform)
data_size = len(train_dataset)

indices = list(range(data_size))

split = int(np.round(0.1*data_size))

train_indices = indices[split:]

valid_indics = indices[:split]

train_sampler = SubsetRandomSampler(train_indices)

valid_sampler = SubsetRandomSampler(valid_indics)
train_loader = DataLoader(train_dataset,batch_size=32,sampler=train_sampler)

valid_loader = DataLoader(train_dataset,batch_size=32,sampler=valid_sampler)
model = resnet101(pretrained=False)

model.load_state_dict(torch.load('../input/resnet101/resnet101.pth'))

for param in model.parameters():

    param.require_grad = False

    

    

model.avg_pool = nn.AdaptiveAvgPool2d(1)

model.fc = nn.Sequential(

              nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

              nn.Dropout(p=0.25),

              nn.Linear(in_features=2048, out_features=2048, bias=True),

              nn.ReLU(),

              nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

              nn.Dropout(p=0.5),

              nn.Linear(in_features=2048, out_features=5, bias=True),

             )

fc_parameters = model.fc.parameters()

for param in fc_parameters:

    param.require_grad = True

model = model.cuda()

device = torch.device("cuda:0")
criteria = nn.CrossEntropyLoss()

plist = [

        {'params': model.layer4.parameters(), 'lr': 1e-5}

        ]

optimizer_ft = optim.Adam(plist, lr=0.001)

lr_sch = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

def fit(epochs,model,optimizer,criteria, scheduler):

    for epoch in range(epochs+1):

        training_loss = 0.0

        validation_loss = 0.0

        correct = 0.0

        total = 0

        print('{}/{} Epochs'.format(epoch+1,epochs))

        

        model.train()

        for batch_idx,d in enumerate(train_loader):

            data = d['images'].cuda()

            target = d['labels'].cuda()

            

            optimizer.zero_grad()

            output = model(data)

            loss = criteria(output,target)

            loss.backward()

            optimizer.step()

            

            training_loss = training_loss + ((1/(batch_idx+1))*(loss.data-training_loss))

            if batch_idx%20==0:

                print('Training loss {}'.format(training_loss))

            pred = output.data.max(1,keepdim=True)[1]

            correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())

            total +=data.size(0)

            print('Accuracy on batch {} on Training is {}'.format(batch_idx,(100*correct/total)))

            

        model.eval()

        for batch_idx ,d in enumerate(valid_loader):

            data = d['images'].cuda()

            target = d['labels'].cuda()

            

            output = model(data)

            loss = criteria(output,target)

            

            validation_loss = validation_loss +((1/(batch_idx+1))*(loss.data-validation_loss))

            if batch_idx%20==0:

                print('Validation_loss {}'.format(validation_loss))

            pred = output.data.max(1,keepdim=True)[1]

            correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())

            total+=data.size(0)

            print('Validation Accuracy on Batch {} is {}'.format(batch_idx,(100*correct/total)))

            

    return model
fit(10,model,optimizer_ft,criteria, lr_sch)
class Prediction(Dataset):

    def __init__(self,csv,transform):

        self.data = pd.read_csv(csv)

        self.transform = transform

        

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self,idx):

        image_path = os.path.join('../input/aptos2019-blindness-detection/test_images/'+self.data.loc[idx]['id_code']+'.png')

        image = Image.open(image_path)

        image = self.transform(image)

        return {'images':image}
test_dataset = Prediction('../input/aptos2019-blindness-detection/test.csv',data_transform)
test_loader = DataLoader(test_dataset)
prediction = []

for batch_idx,d in enumerate(test_loader):

    data = d['images'].cuda()

    output = model(data)

    output = output.cpu().detach().numpy()

    prediction.append(np.argmax(output))
submission = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

submission.head()
submission['diagnosis'] = prediction

submission.head()
submission.to_csv('submission.csv',index = False)