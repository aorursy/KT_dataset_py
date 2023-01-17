# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

from __future__ import print_function, division

import os



import pdb



import argparse

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from PIL import Image

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, utils





class TestDatasets(Dataset):



        def __init__(self,  test_dir, transform=None):

            self.root_dir=test_dir

            self.transform = transform

            

            self.abs_paths=self.get_paths()

    

        def __len__(self):

            return len(self.abs_paths)

            



        def get_paths(self):

            self.paths_list = os.listdir(self.root_dir)

            abs_paths_list=[]

            for path in self.paths_list:

                abs_path=self.root_dir+path

                abs_paths_list.append(abs_path)

            return abs_paths_list

        

        def __getitem__(self, idx):

            img_name = self.abs_paths[idx]

            id=self.paths_list[idx]

            image = Image.open(img_name)

            

            if self.transform:

                image = self.transform(image)

            return image,id















  



class CactusDatasets(Dataset):

    """Face Landmarks dataset."""



    def __init__(self, csv_file, root_dir, transform=None,test=False):

        self.test=test

        self.cactus_frame = pd.read_csv(csv_file)

        self.root_dir = root_dir

        self.transform = transform



    def __len__(self):

        return len(self.cactus_frame)



    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir,self.cactus_frame.iloc[idx, 0])

        image = Image.open(img_name)



        if self.transform:

            image = self.transform(image)

        if not self.test:



            label = self.cactus_frame.iloc[idx, 1].astype('float32')

            return image,label

        else:

            return image





transform = transforms.Compose([transforms.Resize(size=(32, 32), interpolation=2),

                                transforms.RandomHorizontalFlip(),

                                transforms.RandomVerticalFlip(),

                                transforms.ToTensor()])





tr_cactus_datasets=CactusDatasets(csv_file='../input/train.csv',

                                 root_dir='../input/train/train',

                                 transform=transform)





test_datasets=TestDatasets('../input/test/test/',transform=transform)   







class Net(nn.Module):

    def __init__(self):

        super(Net,self).__init__()

        self.conv1=nn.Conv2d(3,6,5,1)

        self.conv2=nn.Conv2d(6,16,5,1)

        self.bn1=nn.BatchNorm2d(6)

        self.bn2=nn.BatchNorm2d(16)

        self.fc1=nn.Linear(5*5*16,120)

        self.fc2=nn.Linear(120,84)

        self.fc3=nn.Linear(84,1)

        self.drop1=nn.Dropout2d(p=0.5)

    def forward(self,x ):

        x=self.conv1(x)

        x=F.relu(self.bn1(x))

        x=F.max_pool2d(x,2,2)

        x=self.conv2(x)

        x=F.relu(self.bn2(x))

        x=F.max_pool2d(x,2,2)

        x=x.view(-1,5*5*16)

        x=F.relu(self.fc1(x))

        x=F.relu(self.fc2(x))

        x=self.fc3(self.drop1(x))

        x=F.sigmoid(x)

        return x









def train(model, device, train_loader, optimizer, epoch):

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)



        loss=F.binary_cross_entropy(output,target)

        loss.backward()

        optimizer.step()

        if batch_idx % 10 == 0:

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

                epoch, batch_idx * len(data), len(train_loader.dataset),

                100. * batch_idx / len(train_loader), loss.item()))





            



            

def main():

    torch.manual_seed(1)

    use_cuda=True

    epoches=20

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}



    

    train_loader = torch.utils.data.DataLoader(tr_cactus_datasets, batch_size=64, shuffle=True,**kwargs)



    test_loader=torch.utils.data.DataLoader(test_datasets,batch_size=4000, shuffle=False,**kwargs) 

    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Net().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, epoches+ 1):

        train(model, device, train_loader, optimizer, epoch)

     

    model.eval()

    predicts=[]

    ids=[]

    with torch.no_grad():

        for data,img_name in test_loader:

            data=data.to(device)          

            preds=model(data)

            preds=list(np.squeeze(preds.cpu().numpy()))

            ids=img_name



    df=pd.DataFrame({'id':ids,'has_cactus':preds})

    df.to_csv('submission.csv ')

    



        

if __name__ == '__main__':

    main()









# Any results you write to the current directory are saved as output.
import pandas as pd

import os

print(os.listdir('./'))

df=pd.read_csv('./sample_submission.csv')

print(df)

import os

print(os.listdir('./'))