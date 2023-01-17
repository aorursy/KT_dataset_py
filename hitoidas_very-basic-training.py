# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os





# Any results you write to the current directory are saved as output.
import torch.utils.data as D

from PIL import Image

from torchvision import transforms,models



class ImagesDS(D.Dataset):

    def __init__(self, df, site=1, channels=[1,2,3,4,5,6]):

        self.records = df.to_records(index=False)

        self.channels = channels

        self.site = site

        self.img_dir = '../input'

        self.len = df.shape[0]

        

    @staticmethod

    def _load_img_as_tensor(file_name):

        with Image.open(file_name) as img:

            img = transforms.Resize((224,224))(img)

            img = transforms.CenterCrop(224)(img)

            return transforms.ToTensor()(img)



    def _get_img_path(self, index, channel):

        experiment, well, plate = self.records[index].experiment, self.records[index].well, self.records[index].plate

        return '/'.join([self.img_dir,'train',experiment,f'Plate{plate}',f'{well}_s{self.site}_w{channel}.png'])

        

    def __getitem__(self, index):

        paths = [self._get_img_path(index, ch) for ch in self.channels]

        img = torch.cat([self._load_img_as_tensor(img_path) for img_path in paths])

        return img, int(self.records[index].sirna)



    def __len__(self):

        return self.len

        
df = pd.read_csv('/kaggle/input/train.csv')

train1 = ImagesDS(df)

train2 = ImagesDS(df,site=2)

trainset = D.ConcatDataset([train1,train2])

from torch.utils.data.sampler import SubsetRandomSampler

import torch



dataset_size = len(trainset)

indices = list(range(dataset_size))

split = int(np.floor(0.2 * dataset_size))

np.random.seed(42)

np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]



# Creating PT data samplers and loaders:

train_sampler = SubsetRandomSampler(train_indices)

valid_sampler = SubsetRandomSampler(val_indices)



train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, 

                                           sampler=train_sampler,num_workers=4)

val_loader = torch.utils.data.DataLoader(trainset, batch_size=32,

                                                sampler=valid_sampler,num_workers=4)
from torch import optim,nn

model = models.densenet201(pretrained=True)

model.features.conv0 = nn.Conv2d(6,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)

model.classifier = nn.Sequential(nn.BatchNorm1d(1920),

                                nn.Linear(1920,1108),

                                nn.LogSoftmax(dim=1))

device = torch.device("cuda:0")

model.to(device)
from sklearn.metrics import accuracy_score

optimizer = optim.Adam(model.parameters(),lr=0.003)

criterion = nn.CrossEntropyLoss()

epochs = 13

for i in range(epochs):

    tr_loss = 0

    v_loss = 0

    for im,y in train_loader:

        im,y = im.to(device),y.to(device)

        optimizer.zero_grad()

        output = model(im)

        p_y = torch.exp(output)

        loss = criterion(output,y)

        loss.backward()

        optimizer.step()

        tr_loss+=loss.item()

    with torch.no_grad():

        model.eval()

        for im,y in val_loader:

            im,y=im.to(device),y.to(device)

            output = model(im)

            p_y = torch.exp(output)

            loss = criterion(output,y)

            v_loss+=loss.item()

        model.train()

    print(f'Epoch:{i+1}  TrainLoss;{tr_loss/len(train_loader)}  ValLoss:{v_loss/len(val_loader)}')

    #print(f'TrainAccuracy:{acc} ValAcc:{vacc}')

    print('---------------------------------------------------------------------------------------------')

    

torch.save(model,'./model.pth')



        

    