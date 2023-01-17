# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

'''

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

'''



# Any results you write to the current directory are saved as output.

import cv2

import torch

import torch.nn as nn

import torchvision.models as models

from torch.utils.data import Dataset,DataLoader
class Resize(object):

    

    def __init__(self,_type = 'train'):

        self.path = '/kaggle/input/aptos2019-blindness-detection/'

        

        if 'train' in _type:

            _type = 'train_images'

            self.df = pd.read_csv(os.path.join(self.path,'train.csv'))

            !rm -r resized_train_images

            !mkdir resized_train_images

            self.new_path = 'resized_train_images'

            self.root = os.path.join(self.path,_type)

        elif 'test' in _type:

            _type = 'test_images'

            self.df = os.path.join(self.path,'test.csv')

            !rm -r resized_test_images

            !mkdir resized_test_images

            self.new_path = 'resized_test_images'

            self.root = pd.read_csv(os.path.join(self.path,_type))

        else:

            raise print(f'type should contain either train or test but got {_type}')

            

    def __call__(self):

        fnames = self.df.set_index('id_code').index.values

        

        for fname in fnames:

            #read -> convert BGR to RGB

            Image = cv2.cvtColor(cv2.imread(os.path.join(self.root,fname+'.png')),cv2.COLOR_BGR2RGB)

            ## resize 224x224 compatible with resnet-like structures.

            Image = cv2.resize(Image,(224,224))

            result = cv2.imwrite(os.path.join(self.new_path,fname+'.png'), Image)

            print(f'saved successfully at {self.new_path}')
resize = Resize(_type='train')

resize()
class EyeBallDataset(Dataset):

    def __init__(self,_type = 'train'):

        self.path = '/kaggle/input/aptos2019-blindness-detection/'

        if _type == 'train':

            print('train dataset')

            self.df = pd.read_csv(os.path.join(self.path,'train.csv'))

            self.root = 'resized_train_images'

        elif _type == 'test':

            print('test dataset')

            self.df = pd.read_csv(os.path.join(self.path,'test.csv'))

            self.root = 'resized_test_images'

        else:

            raise print(f'_type should be either train or test but got{_type}')

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self,idx):

        fname = self.df.id_code.values[idx]

        label = self.df.diagnosis.values[idx]

        Image = cv2.imread(os.path.join(self.root,fname+'.png'))

        return torch.tensor(Image),label
train_images = EyeBallDataset('train')

train_loader = DataLoader(train_images, batch_size=25, shuffle=True)
classes = {0:'No DR', 1:'Mild', 2:'Moderate', 3:'Severe', 4:'Proliferative Dr'}
import matplotlib.pyplot as plt

a = next(iter(train_loader))



fig,ax = plt.subplots(5,5,figsize=(25,25))



for i in range(25):

    j = i//5

    k = i%5

    ax[j,k].imshow(a[0][i])

    ax[j,k].set_title(f'{classes[int(a[1][i])]}',fontsize=20)
class ResNet101(nn.Module):

    def __init__(self,pretrained=True):

        super().__init__()

        encoder = models.resnet101(pretrained=pretrained)

        encoder = nn.Sequential(*list(encoder.children()))

    

        self.cnn = nn.Sequential(

            encoder[0],

            encoder[1],

            encoder[2],

            encoder[3],

            encoder[4],

            encoder[5],

            encoder[6],

            encoder[7],

            encoder[8]

        )

        self.clf = nn.Linear(2048,5,bias=False)

    def forward(self,x):

        x = self.cnn(x)

        x = x.view(x.size(0),-1)

        x = self.clf(x)

        return x
train_images = EyeBallDataset('train')

train_loader = DataLoader(train_images, batch_size=25, shuffle=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ResNet101().to(device)
accs = []

losses = []

epochs = 10

lr = 1e-3

optimizer = torch.optim.Adam(model.parameters(),lr = lr)

criterion = nn.CrossEntropyLoss(reduction='mean')
from tqdm import tqdm_notebook as tqdm

model.train()

for epoch in range(epochs):

    print(f'epochs {epoch+1}/{epochs}')

    running_loss = 0.0

    running_acc = 0.0

    for idx, (inputs,labels) in tqdm(enumerate(train_loader),total=len(train_loader)):

        optimizer.zero_grad()

        inputs = inputs.to(device)

        labels = labels.to(device)

        outputs = model(inputs.permute(0,3,2,1).float())

        loss = criterion(outputs,labels)

        loss.backward()

        optimizer.step()

        running_loss += loss

        running_acc += (outputs.argmax(1)==labels).float().mean()

    print('loss : {:.4f} acc: {:.2f}'.format(running_loss/len(train_loader),running_acc/len(train_loader)))

    losses.append(running_loss/len(train_loader))

    accs.append(running_acc/len(train_loader))
figs , ax = plt.subplots(1,2,figsize=(20,5))

ax[0].plot(losses)

ax[0].set_title('train_loss')

ax[1].plot(accs)

ax[1].set_title('train_acc')
!rm -r resized_train_images

!rm -r resized_test_images