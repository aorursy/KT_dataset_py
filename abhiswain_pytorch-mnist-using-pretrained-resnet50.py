# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import seaborn as sns 

import matplotlib.pyplot as plt

import torch

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, utils, models

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
class MNIST_data(Dataset):

    """MNIST dtaa set"""

    

    def __init__(self, file_path, 

                 transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), 

                     transforms.Normalize(mean=(0.5,), std=(0.5,))])

                ):

        

        df = pd.read_csv(file_path)

        

        if len(df.columns) == 784:

            # test data

            self.X = df.values.reshape((-1,28,28)).astype(np.uint8)[:,:,:,None]

            self.y = None

        else:

            # training data

            self.X = df.iloc[:,1:].values.reshape((-1,28,28)).astype(np.uint8)[:,:,:,None]

            self.y = torch.from_numpy(df.iloc[:,0].values)

            

        self.transform = transform

    

    def __len__(self):

        return len(self.X)



    def __getitem__(self, idx):

        if self.y is not None:

            return self.transform(self.X[idx]), self.y[idx]

        else:

            return self.transform(self.X[idx])
batch_size = 64



train_dataset = MNIST_data('/kaggle/input/digit-recognizer/train.csv', transform= transforms.Compose(

                            [transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]))

test_dataset = MNIST_data('/kaggle/input/digit-recognizer/test.csv')





train_loader = torch.utils.data.DataLoader(dataset=train_dataset,

                                           batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,

                                           batch_size=batch_size, shuffle=False)
model = models.resnet50(pretrained=True)
model
def change_layers(model):

    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    model.fc = nn.Linear(2048, 10, bias=True)

    return model
change_layers(model)


if(torch.cuda.is_available()):

    model = model.cuda()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
import time



print('Training....')

total = 0

correct = 0

start = time.time()



for epoch in range(10):

    

    for i, data in enumerate(train_loader, 1):

        images, labels = data



        if(torch.cuda.is_available()):

            images = images.cuda()

            labels = labels.cuda()



        optimizer.zero_grad()    

        outputs = model(images)



        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()



        loss = criterion(outputs, labels)

        if(i%100 == 0):

            print('Epoch: {} Batch: {} loss: {}'.format(epoch, i, loss.item()))



        loss.backward()

        optimizer.step()



print('Training Completed in: {} secs'.format(time.time()-start))

print('Training accuracy: {} %'.format((correct/total)*100))



        
print('Predicting....')

start = time.time()



predictions = torch.LongTensor()

for i, data in enumerate(test_loader, 1):

    data = data.cuda()

    

    if(i%100 == 0):

        print('Batch {} done'.format(i))

    outputs = model(data)

    

    pred = outputs.cpu().data.max(1, keepdim=True)[1]

    predictions = torch.cat((predictions, pred), dim=0)

    

print('Completed in {} secs'.format(time.time() - start))
out_df = pd.DataFrame(np.c_[np.arange(1, len(test_dataset)+1)[:,None], predictions.numpy()], columns=['ImageId', 'Label'])
out_df.to_csv('submission.csv', index=False)
out_df.head()