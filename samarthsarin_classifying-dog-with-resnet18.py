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

from torchvision.models import resnet18

import torch.nn as nn

import torchvision.transforms as transforms

import torch.optim as optim

from torch.utils.data import DataLoader,Dataset

from torch.utils.data.sampler import SubsetRandomSampler

from PIL import Image

from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm
le = LabelEncoder()

df = pd.read_csv('../input/dog-breed-identification/labels.csv')

df.head()

df['breed'] = le.fit_transform(df['breed'])
df.head()
class DogDataset(Dataset):

    def __init__(self,csv,transform):

        self.data = csv

        self.transform = transform

        self.labels = torch.eye(120)[self.data['breed']]

        

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self,idx):

        image_path = os.path.join('../input/dog-breed-identification/train/'+self.data.loc[idx]['id']+'.jpg')

        image = Image.open(image_path)

        image = self.transform(image)

        label = torch.tensor(self.data.loc[idx]['breed'])

        return {'images':image,'labels':label}
simple_transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([0.496,0.456,0.406],[0.229,0.224,0.225])])
train_dataset = DogDataset(df,simple_transform)
data_size = len(train_dataset)

indicies = list(range(data_size))

split = int(np.round(0.2*data_size,0))

training_indicies = indicies[split:]

validation_indices = indicies[:split]

train_sampler = SubsetRandomSampler(training_indicies)

valid_sampler = SubsetRandomSampler(validation_indices)
train_loader = DataLoader(train_dataset,batch_size=32,sampler=train_sampler)

valid_loader = DataLoader(train_dataset,batch_size=32,sampler=valid_sampler)
model = resnet18(pretrained=False)

model.load_state_dict(torch.load('../input/resnet18/resnet18.pth'))

for param in model.parameters():

    param.require_grad = False

model.fc = nn.Linear(512,120)

fc_parameters = model.fc.parameters()

for param in fc_parameters:

    param.require_grad = True

model = model.cuda()
criteria= nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(),lr = 0.001)
def fit(epochs,model,optimizer,criteria):

    for epoch in range(epochs):

        training_loss = 0.0

        validation_loss = 0.0

        correct = 0

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

            

            training_loss = training_loss + (1/(batch_idx+1)*(loss.data-training_loss))

            if batch_idx%20==0:

                print('Training Loss is {}'.format(training_loss))

                

            pred = output.data.max(1,keepdim = True)[1]

            correct +=np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())

            total +=data.size(0)

            print('Batch ID {} is having training Accuracy of {}'.format(batch_idx,100*correct/total))

            

        model.eval()

        for batch_idx,d in enumerate(valid_loader):

            data = d['images'].cuda()

            target = d['labels'].cuda()

            

            output = model(data)

            loss = criteria(output,target)

            validation_loss = validation_loss + ((1)/(batch_idx+1)*(loss.data-validation_loss))

            

            if batch_idx % 20==0:

                print('Validation Loss is {}'.format(validation_loss))

            pred = output.data.max(1,keepdim=True)[1]

            correct+= np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())

            total += data.size(0)

            print('Batch id {} is having Validation Accuracy of {}'.format(batch_idx,100*correct/total))

            

    return model
fit(10,model,optimizer,criteria)
sample = pd.read_csv('../input/dog-breed-identification/sample_submission.csv')

sample.head()
class Prediction(Dataset):

    def __init__(self,csv,transform):

        self.data = pd.read_csv(csv)

        self.transform = transform

        

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self,idx):

        image_path = os.path.join('../input/dog-breed-identification/test/'+self.data.loc[idx]['id']+'.jpg')

        image = Image.open(image_path)

        image = self.transform(image)

        return {'images':image}
test_dataset = Prediction('../input/dog-breed-identification/sample_submission.csv',simple_transform)
test_loader = DataLoader(test_dataset)
predict = []

for batch_idx, d in enumerate(test_loader):

    data = d['images'].cuda()

    output = model(data)

    output = output.cpu().detach().numpy()

    predict.append(list(output[0]))

    
for i in tqdm(range(len(predict))):

    sample.iloc[i,1:]  = predict[i]
sample.head()
sample.to_csv('sample_submission.csv',index = False)