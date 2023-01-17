## This Python 3 environment comes with many helpful analytics libraries installed

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

import torch.nn.functional as F

from torch.utils.data import Dataset,DataLoader

from tqdm import tqdm_notebook as tqdm
class InceptionBranch(nn.Module):

    

    def __init__(self,in_channels,out_1x1,out_3x3_1,out_3x3_2,out_5x5_1,out_5x5_2,out_pool):

        super().__init__()

        self.branch1 = nn.Sequential(

            nn.Conv2d(in_channels,out_1x1, kernel_size=1,stride=1,padding=0),

            nn.BatchNorm2d(out_1x1),

            nn.ReLU(True)

        )

        self.branch2 = nn.Sequential(

            nn.Conv2d(in_channels,out_3x3_1, kernel_size=1,stride=1,padding=0),

            nn.BatchNorm2d(out_3x3_1),

            nn.ReLU(True),

            nn.Conv2d(out_3x3_1,out_3x3_2, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(out_3x3_2),

            nn.ReLU(True)

        )

        self.branch3 = nn.Sequential(

            nn.Conv2d(in_channels,out_5x5_1,kernel_size=1,stride=1,padding=0),

            nn.BatchNorm2d(out_5x5_1),

            nn.ReLU(True),

            nn.Conv2d(out_5x5_1,out_5x5_2,kernel_size=5,stride=1,padding=2),

            nn.BatchNorm2d(out_5x5_2),

            nn.ReLU(True)

        )

        self.branch4 = nn.Sequential(

            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),

            nn.Conv2d(in_channels,out_pool,kernel_size=1, stride=1,padding=0),

            nn.BatchNorm2d(out_pool),

            nn.ReLU(True)

        )

        

    def forward(self,x):

        return torch.cat([self.branch1(x),self.branch2(x),self.branch3(x),self.branch4(x)],1)
class Googlenet(nn.Module):

    def __init__(self,n_classes):

        super().__init__()

        self.pre_layer = nn.Sequential(

            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3),

            nn.ReLU(True),

            nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True),

            nn.CrossMapLRN2d(5,1e-3,0.75,1),

            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=1,stride=1,padding=0),

            nn.ReLU(True),

            nn.Conv2d(in_channels=64,out_channels=192,kernel_size=3,stride=1,padding=1),

            nn.ReLU(True),

            nn.CrossMapLRN2d(5,1e-3,0.75,1),

            nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True)

        )

            

        self.block1 = InceptionBranch(192,64,96,128,16,32,32)

        self.block2 = InceptionBranch(256,128,128,192,32,96,64)

        self.block3 = InceptionBranch(480,192,96,208,16,48,64)

        self.block4 = InceptionBranch(512,160,112,224,24,64,64)

        self.block5 = InceptionBranch(512,128,128,256,24,64,64)

        self.block6 = InceptionBranch(512,112,144,288,32,64,64)

        self.block7 = InceptionBranch(528,256,160,320,32,128,128)

        self.block8 = InceptionBranch(832,256,160,320,32,128,128)

        self.block9 = InceptionBranch(832,384,192,384,48,128,128)

        self.fc = nn.Linear(1024,n_classes)

            

    def forward(self,x):

        x = self.pre_layer(x)

        x = self.block1(x)

        x = self.block2(x)

        x = F.max_pool2d(x,kernel_size=2,stride=2,ceil_mode=True)

        x = self.block3(x)

        x = self.block4(x)

        x = self.block5(x)

        x = self.block6(x)

        x = self.block7(x)

        x = F.max_pool2d(x,kernel_size=2,stride=2,ceil_mode=True)

        x = self.block8(x)

        x = self.block9(x)

        x = F.avg_pool2d(x,kernel_size=2,stride=1,ceil_mode=True)

        x = x.view(x.size(0),-1)

        x = self.fc(x)

        return x
!pip install torchsummary

model = Googlenet(2)

from torchsummary import summary

summary(model,(3, 56, 56))
class Resize(object):

    

    def __call__(self,resize):

        records = {}

        root = '/kaggle/input/plant-seedlings-classification/'

        for _type in ['train','test']:

            path = os.path.join(root,_type)

            if _type == 'train':

                !rm -r resized_train

                !mkdir resized_train

                classes = os.listdir(path)

                for _class in classes:

                    print(f'resized {_class}')

                    path1 = os.path.join(path,_class)

                    os.mkdir(f'resized_train/{_class}')

                    for root,_,fnames in sorted(os.walk(path1,followlinks=True)):

                        for fname in fnames:

                            if '.png' in fname:

                                image = cv2.imread(os.path.join(root,fname))

                                try:

                                    image = cv2.resize(image,(resize,resize))

                                    result = cv2.imwrite(f'resized_train/{_class}/{fname}',image)

                                    print(f'train successfully saved at resized_train/{_class}/{fname}')

                                    records[f'resized_train/{_class}/{fname}'] = f'{_class}'

                                except:

                                    raise print('got {} which is invalid'.format(resize))

                records = pd.DataFrame.from_dict(records,orient='index').reset_index().rename(columns={'index':'fname',0:'label'})

                return records

            else:

                !rm -r resized_test

                !mkdir resized_test

                print('resize test')

                for root,_,fnames in sorted(os.walk(path,followlinks=True)):

                    for fname in fnames:

                        if '.png' in fname:

                            image = cv2.imread(os.path.join(root,fname))

                            try:

                                image = cv2.resize(image,(resize,resize))

                                result = cv2.imwrite(f'resized_test/{fname}', image)

                                print(f'test :{fname} successfully saved {result}')

                            except:

                                print('got {} which is invalid'.format(resize))

                            

                                

                        
resize = Resize()

train = resize(56)
seedling_num_dict = {idx:i for idx,i in enumerate(train.label.unique())}

seedling_dict = {i:idx for idx,i in enumerate(train.label.unique())}
train['num_label'] = train.label.map(seedling_dict)
seedling_dict
class SeedlingDataset(Dataset):

    def __init__(self,df,_type='train'):

        if _type=='train':

            self.df = df

            self.root = 'resized_train'

        elif _type=='test':

            self.root = 'resized_test'

        else:

            raise print(f'shoud be either train or test but got {_type}')

    def __len__(self):

        return len(self.df)

    def __getitem__(self,idx):

        fname = self.df.fname.values[idx]

        Image = cv2.imread(fname)

        label = self.df.num_label[idx]

        return Image, label
train_data = SeedlingDataset(train,_type='train')

train_dataloader = torch.utils.data.DataLoader(train_data,shuffle=True,batch_size=6)
import matplotlib.pyplot as plt

a = next(iter(train_dataloader))

fig,ax = plt.subplots(2,3,figsize=(25,25))



for i in range(6):

    j = i//3

    k = i%3

    ax[j,k].imshow(a[0][i])

    ax[j,k].set_title(seedling_num_dict[int(a[1][i])],fontsize=20)

plt.tight_layout()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Googlenet(12).to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

criterion = nn.CrossEntropyLoss()

epochs = 2

batch_size = 32
losses = []

accs = []

train_data = SeedlingDataset(train,_type='train')

train_loader = torch.utils.data.DataLoader(train_data,shuffle=True,batch_size=batch_size)

for epoch in range(epochs):

    running_loss = 0.0

    running_acc = 0.0

    for idx,(inputs,labels) in tqdm(enumerate(train_loader),total=len(train_loader)):

        

        inputs = inputs.to(device)

        labels = labels.to(device)

        

        outputs = model(inputs.permute(0,3,2,1).float())

        

        loss = criterion(outputs,labels)

        loss.backward()

        optimizer.step()

        running_loss += loss

        running_acc += (torch.argmax(outputs,1).unsqueeze(0)==labels).float().mean()

        running_loss += loss

    print('{}/{} : loss : {:.4f} || acc: {:.2f}%'.format(epoch+1,epochs,running_loss/len(train_loader),running_acc/len(train_loader)))

    losses.append(running_loss/len(train_loader))

    accs.append(running_acc/len(train_loader))
!rm -r resized_train

!rm -r resized_test
fig,ax = plt.subplots(1,2,figsize=(15,5))

ax[0].plot(losses)

ax[0].set_title('loss')

ax[1].plot(accs)

ax[1].set_title('acc')