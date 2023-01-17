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

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset,DataLoader

from torchvision import transforms,models

from tqdm import tqdm_notebook as tqdm

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")
train = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')

data0 = pd.read_feather('/kaggle/usr/lib/resize_and_load_with_feather_format_much_faster/train_data_0.feather')

data1 = pd.read_feather('/kaggle/usr/lib/resize_and_load_with_feather_format_much_faster/train_data_1.feather')

data2 = pd.read_feather('/kaggle/usr/lib/resize_and_load_with_feather_format_much_faster/train_data_2.feather')

data3 = pd.read_feather('/kaggle/usr/lib/resize_and_load_with_feather_format_much_faster/train_data_3.feather')
data_full = pd.concat([data0,data1,data2,data3],ignore_index=True)
class GraphemeDataset(Dataset):

    def __init__(self,df,label,_type='train'):

        self.df = df

        self.label = label

    def __len__(self):

        return len(self.df)

    def __getitem__(self,idx):

        label1 = self.label.consonant_diacritic.values[idx]

        label2 = self.label.grapheme_root.values[idx]

        label3 = self.label.vowel_diacritic.values[idx]

        image = self.df.iloc[idx][1:].values.reshape(64,64).astype(np.uint8)



        return torch.tensor(image/255.0),label1,label2,label3
image = GraphemeDataset(data_full,train)

loader = torch.utils.data.DataLoader(image,batch_size=15,shuffle=True)
fig,ax = plt.subplots(3,5,figsize=(25,25))

ax = ax.flatten()

a = next(iter(loader))

for i in range(15):

    ax[i].imshow(a[0][i],cmap='gray')

    ax[i].set_title([int(a[1][i]),int(a[2][i]),int(a[3][i])])
class BornoBranch(nn.Module):

    def __init__(self,in_channels,out_channels):

        super().__init__()

        self.branch1 = nn.Sequential(

            nn.Conv2d(in_channels,out_channels,kernel_size=5),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(True),

            nn.Conv2d(out_channels,out_channels,kernel_size=3),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(True)

            

        )

        self.branch2 = nn.Sequential(

            nn.Conv2d(in_channels,out_channels,kernel_size=5),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(True),

            nn.Conv2d(out_channels,out_channels,kernel_size=3),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(True)

        )

        

    def forward(self,x):

        x1 = self.branch1(x)

        x2 = self.branch2(x)

        

        return x1+x2
class BornoNet(nn.Module):

    

    def __init__(self):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(1,32,kernel_size=2,padding=2),

            nn.BatchNorm2d(32),

            nn.ReLU(True),

            nn.Conv2d(32,32,kernel_size=2,padding=2),

            nn.BatchNorm2d(32),

            nn.ReLU(True),

        )

        self.branches1 = BornoBranch(32,64)

        self.branches2 = BornoBranch(64,64)

        self.maxpool = nn.MaxPool2d(3)

        self.batch1 = nn.BatchNorm2d(32)

        self.batch2 = nn.BatchNorm2d(64)

        self.batch3 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(1600,7)

        self.fc2 = nn.Linear(1600,168)

        self.fc3 = nn.Linear(1600,11)

    def forward(self,x):

        x = F.dropout(self.conv(x),0.25)

        x = self.batch1(x)

        x = self.branches1(x)

        x = F.dropout(self.maxpool(x),0.25)

        x = self.batch2(x)

        x = self.branches2(x)

        x = F.dropout(self.maxpool(x),0.25)

        x = self.batch2(x)

        x = x.view(x.size(0),-1)

        x1 = self.fc1(x)

        x2 = self.fc2(x)

        x3 = self.fc3(x)

        #Skip Softmax ;-)

        return x1,x2,x3
!pip install torchsummary

from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = BornoNet().to(device)

summary(model, (1, 64, 64))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = BornoNet().to(device)

model.load_state_dict(torch.load('/kaggle/input/grapheme-modified-bornonet-naive-learning-2/saved_weights.pth'))

optimizer = optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.9)

criterion = nn.CrossEntropyLoss()

batch_size=32
epochs = 50

model.train()

losses = []

accs = []

val_losses = []

val_accs = []

for epoch in range(epochs):

    train_index =train.groupby(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']).apply(lambda x: x.sample(5)).image_id.values

    reduced_train = train.loc[train.image_id.isin(train_index)]

    train_data = data_full.loc[data_full.image_id.isin(train_index)]

    

    train_image = GraphemeDataset(train_data,reduced_train)

    train_loader = torch.utils.data.DataLoader(train_image,batch_size=batch_size,shuffle=True)

    

    test_index =train.groupby(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']).apply(lambda x: x.sample(1)).image_id.values

    reduced_test = train.loc[train.image_id.isin(test_index)]

    test_data = data_full.loc[data_full.image_id.isin(test_index)]

    

    test_image = GraphemeDataset(test_data,reduced_test)

    test_loader = torch.utils.data.DataLoader(test_image,batch_size=batch_size,shuffle=True)

    

    print('epochs {}/{} '.format(epoch+1,epochs))

    running_loss = 0.0

    running_acc = 0.0

    for idx, (inputs,labels1,labels2,labels3) in tqdm(enumerate(train_loader),total=len(train_loader)):

        inputs = inputs.to(device)

        labels1 = labels1.to(device)

        labels2 = labels2.to(device)

        labels3 = labels3.to(device)

        

        optimizer.zero_grad()

        outputs1,outputs2,outputs3 = model(inputs.unsqueeze(1).float())

        loss1 = criterion(outputs1,labels1)

        loss2 = criterion(outputs2,labels2)

        loss3 = criterion(outputs3,labels3)

        running_loss += loss1+loss2+loss3

        running_acc += (outputs1.argmax(1)==labels1).float().mean()

        running_acc += (outputs2.argmax(1)==labels2).float().mean()

        running_acc += (outputs3.argmax(1)==labels3).float().mean()

        (loss1+loss2+loss3).backward()

        optimizer.step()

    scheduler.step()

    losses.append(running_loss/len(train_loader))

    accs.append(running_acc/(len(train_loader)*3))

    print('acc : {:.2f}%'.format(100*running_acc/(len(train_loader)*3)))

    print('loss : {:.4f}'.format(running_loss/len(train_loader)))

    

    with torch.no_grad():

        running_loss = 0.0

        running_acc = 0.0

        for idx, (inputs,labels1,labels2,labels3) in tqdm(enumerate(test_loader),total=len(test_loader)):

            

            inputs = inputs.to(device)

            labels1 = labels1.to(device)

            labels2 = labels2.to(device)

            labels3 = labels3.to(device)

            

            outputs1,outputs2,outputs3 = model(inputs.unsqueeze(1).float())

            

            loss1 = criterion(outputs1,labels1)

            loss2 = criterion(outputs2,labels2)

            loss3 = criterion(outputs3,labels3)

            running_loss += loss1+loss2+loss3

            running_acc += (outputs1.argmax(1)==labels1).float().mean()

            running_acc += (outputs2.argmax(1)==labels2).float().mean()

            running_acc += (outputs3.argmax(1)==labels3).float().mean()

            

        

        val_losses.append(running_loss/len(test_loader))

        val_accs.append(running_acc/(len(test_loader)*3))

        print('val_acc : {:.2f}%'.format(100*running_acc/(len(test_loader)*3)))

        print('va_loss : {:.4f}'.format(running_loss/len(test_loader)))

            

torch.save(model.state_dict(), 'saved_weights.pth')
fig,ax = plt.subplots(1,2,figsize=(15,5))

ax[0].plot(losses,label='train')

ax[0].plot(val_losses,label='valid')

ax[0].set_title('loss')

ax[0].legend()

ax[1].plot(accs,label='train')

ax[1].plot(val_accs,label='valid')

ax[1].set_title('acc')

ax[0].legend()