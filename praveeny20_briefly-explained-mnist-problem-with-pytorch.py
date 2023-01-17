# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import torch

import torchvision

import torchvision.transforms as transforms

from torch.utils.data import DataLoader, TensorDataset

from torchvision.utils import make_grid



#neural net imports

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.autograd import Variable



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import os

import math

%matplotlib inline
print(torch.cuda.is_available())

print(torch.backends.cudnn.enabled)



if torch.cuda.is_available():

    device = torch.device('cuda')

print(device)
path='../input/digit-recognizer/'
train_df=pd.read_csv(path+'train.csv')

test_df=pd.read_csv(path+'test.csv')
train_df.describe()
test_df.describe()
train_images=(train_df.iloc[:,1:].values).astype('float32')

test_images=(test_df.iloc[:,:].values).astype('float32')

train_labels=train_df['label'].values
train_images,val_images,train_labels,val_labels=train_test_split(train_images,train_labels,stratify=train_labels,random_state=10,test_size=0.2)
train_images = train_images.reshape(train_images.shape[0], 28, 28)

val_images = val_images.reshape(val_images.shape[0], 28, 28)

test_images = test_images.reshape(test_images.shape[0], 28, 28)
plt.imshow(train_images[3],cmap='gray')
plt.imshow(test_images[3],cmap='gray')
train_images_tensor = torch.tensor(train_images)/255.0

train_labels_tensor = torch.tensor(train_labels)

train_tensor = TensorDataset(train_images_tensor, train_labels_tensor)
val_images_tensor = torch.tensor(val_images)/255.0

val_labels_tensor = torch.tensor(val_labels)

val_tensor = TensorDataset(val_images_tensor, val_labels_tensor)
test_images_tensor = torch.tensor(test_images)/255.0
train_loader = DataLoader(train_tensor, batch_size=16, num_workers=2, shuffle=True)

val_loader = DataLoader(val_tensor, batch_size=16, num_workers=2, shuffle=True)

test_loader = DataLoader(test_images_tensor, batch_size=16, num_workers=2, shuffle=False)
class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        

        self.conv_block = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2) 

        )

        

        self.linear_block = nn.Sequential(

            nn.Dropout(p=0.5),

            nn.Linear(128*7*7, 128),

            nn.BatchNorm1d(128),

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Linear(128, 64),

            nn.BatchNorm1d(64),

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Linear(64, 10)

        )

        

    def forward(self, x):

        x = self.conv_block(x)

        x = x.view(x.size(0), -1)

        x = self.linear_block(x)

        

        return x

conv_model = Net()

conv_model
optimizer = optim.Adam(params=conv_model.parameters(), lr=0.003)

criterion = nn.CrossEntropyLoss()



exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)



if torch.cuda.is_available():

    conv_model = conv_model.cuda()

    criterion = criterion.cuda()
def train_model(num_epoch):

    conv_model.train()

    exp_lr_scheduler.step()

    

    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.unsqueeze(1)

        data, target = data, target

        

        if torch.cuda.is_available():

            data = data.cuda()

            target = target.cuda()

            

        optimizer.zero_grad()

        output = conv_model(data)

        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

        

        if (batch_idx + 1)% 100 == 0:

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

                num_epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),

                100. * (batch_idx + 1) / len(train_loader), loss.item()))

            

def evaluate(data_loader):

    conv_model.eval()

    loss = 0

    correct = 0

    

    for data, target in data_loader:

        data = data.unsqueeze(1)

        data, target = data, target

        

        if torch.cuda.is_available():

            data = data.cuda()

            target = target.cuda()

        

        output = conv_model(data)

        

        loss += F.cross_entropy(output, target, size_average=False).data.item()



        pred = output.data.max(1, keepdim=True)[1]

        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        

    loss /= len(data_loader.dataset)

        

    print('\nAverage Val Loss: {:.4f}, Val Accuracy: {}/{} ({:.3f}%)\n'.format(

        loss, correct, len(data_loader.dataset),

        100. * correct / len(data_loader.dataset)))

    
num_epochs = 25



for n in range(num_epochs):

    train_model(n)

    evaluate(val_loader)

def make_predictions(data_loader):

    conv_model.eval()

    test_preds = torch.LongTensor()

    

    for i, data in enumerate(data_loader):

        data = data.unsqueeze(1)

        

        if torch.cuda.is_available():

            data = data.cuda()

            

        output = conv_model(data)

        

        preds = output.cpu().data.max(1, keepdim=True)[1]

        test_preds = torch.cat((test_preds, preds), dim=0)

        

    return test_preds
test_set_preds = make_predictions(test_loader)
submission_df = pd.read_csv(path+"sample_submission.csv")
submission_df['Label'] = test_set_preds.numpy().squeeze()

submission_df.head()
submission_df.to_csv('submission.csv', index=False)
submission_df
