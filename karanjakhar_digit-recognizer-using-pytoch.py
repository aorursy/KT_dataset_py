import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader

from torchvision import models

import torch

from torch import nn

import matplotlib.pyplot as plt

import math
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')
train.head()
test.head()
x_train = np.array(train.iloc[:,1:]).reshape(-1,1,28,28)

x_test = np.array(test).reshape(-1,1,28,28)
y_train = np.array(train['label'])
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train)
class digitdata():

    def __init__(self,x_train,y_train=None,batch_size=64):

        self.x_train = x_train

        self.y_train = y_train

        self.batch_size = batch_size

    def __iter__(self):

        for i in range(self.__len__()):

            yield self.__getitem__(i)

    def __len__(self):

        return math.ceil(len(self.x_train)/self.batch_size)

    

    def __getitem__(self,index):

        if self.y_train is None:

            return torch.tensor(self.x_train[index*self.batch_size:index*self.batch_size+self.batch_size]/255).float()

        else:

            return torch.tensor(self.x_train[index*self.batch_size:index*self.batch_size+self.batch_size]/255).float(),torch.tensor(self.y_train[index*self.batch_size:index*self.batch_size+self.batch_size]).long()
class build_model(nn.Module):

    def __init__(self):

        super(build_model,self).__init__()

        self.pretrained = models.resnet34(pretrained=True)

        self.conv1 = nn.Conv2d(1,16,kernel_size=3,stride=1,padding=0)

        self.conv2 = nn.Conv2d(16,32,kernel_size=3,stride=1,padding=0)

        self.conv3 = nn.Conv2d(32,64,kernel_size=3,stride=1,padding=0)

        self.fc1 = nn.Linear(64,128)

        self.outl = nn.Linear(128,10)

        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=2)

        

    def forward(self,x):

        x = self.conv1(x)

        x = self.relu(x)

        #print(x.size())

        x = self.maxpool(x)

        x = self.conv2(x)

        x = self.relu(x)

        x = self.maxpool(x)

        x = self.conv3(x)

        x = self.relu(x)

        x = self.maxpool(x)

        x = x.view(x.size(0),-1)

        x = self.fc1(x)

        x = self.outl(x)

        return x



        

        
model = build_model()
train_loader = digitdata(x_train,y_train)

test_loader = digitdata(x_test)

val_loader = digitdata(x_val,y_val)
#let's look at the data loaded

x,y = train_loader[0]

print(x.shape,y.shape)

plt.imshow(x[0].reshape(28,28),cmap = 'gray')

plt.show()
def train(epoch):

    total_loss = 0

    model.train()

    for x,target_y in train_loader:

        optimizer.zero_grad()

        output_y = model(x)

        loss = loss_function(output_y,target_y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f'{epoch} Training loss : {total_loss/len(train_loader)}',end=' ')

        
def val():

    total_loss = 0

    accuracy = 0

    model.eval()

    with torch.no_grad():

        for x,target_y in val_loader:

            output_y = model(x)

            loss = loss_function(output_y,target_y)

            total_loss += loss.item()

            accuracy += (torch.max(output_y,1)[1] == target_y).float().sum()

        print(f' | Validation loss : {total_loss/len(val_loader)} | Validation Accuracy : {(accuracy*100)/(len(val_loader)*64)}')
epochs = 10

optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

loss_function = nn.CrossEntropyLoss()



for epoch in range(epochs):

    train(epoch)

    val()
def submission():

    all_sub_res = []

    model.eval()

    with torch.no_grad():

        for x in test_loader:

            output_y = model(x)

            all_sub_res += torch.max(output_y,1)[1].tolist()

    return all_sub_res





        
res = submission()
len(res)
sample_sub = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
sample_sub.head()
len(sample_sub)
sample_sub['Label'] = res
sample_sub.head()
sample_sub.to_csv('submission.csv',index=False)