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

import cv2

import torch

import torch.nn as nn

from torch.utils.data import Dataset,DataLoader

from torchvision import transforms,models

from tqdm import tqdm_notebook as tqdm
test = pd.read_csv('/kaggle/input/bengaliai-cv19/test.csv')
class GraphemeDataset(Dataset):

    def __init__(self,df,_type='train'):

        self.df = df

    def __len__(self):

        return len(self.df)

    def __getitem__(self,idx):

        image = self.df.iloc[idx][1:].values.reshape(64,64).astype(float)

        return image
class ResidualBlock(nn.Module):

    def __init__(self,in_channels,out_channels,stride=1,kernel_size=3,padding=1,bias=False):

        super(ResidualBlock,self).__init__()

        self.cnn1 =nn.Sequential(

            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(True)

        )

        self.cnn2 = nn.Sequential(

            nn.Conv2d(out_channels,out_channels,kernel_size,1,padding,bias=False),

            nn.BatchNorm2d(out_channels)

        )

        if stride != 1 or in_channels != out_channels:

            self.shortcut = nn.Sequential(

                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),

                nn.BatchNorm2d(out_channels)

            )

        else:

            self.shortcut = nn.Sequential()

            

    def forward(self,x):

        residual = x

        x = self.cnn1(x)

        x = self.cnn2(x)

        x += self.shortcut(residual)

        x = nn.ReLU(True)(x)

        return x
class ResNet18(nn.Module):

    def __init__(self):

        super(ResNet18,self).__init__()

        

        self.block1 = nn.Sequential(

            nn.Conv2d(1,64,kernel_size=2,stride=2,padding=3,bias=False),

            nn.BatchNorm2d(64),

            nn.ReLU(True)

        )

        

        self.block2 = nn.Sequential(

            nn.MaxPool2d(1,1),

            ResidualBlock(64,64),

            ResidualBlock(64,64,2)

        )

        

        self.block3 = nn.Sequential(

            ResidualBlock(64,128),

            ResidualBlock(128,128,2)

        )

        

        self.block4 = nn.Sequential(

            ResidualBlock(128,256),

            ResidualBlock(256,256,2)

        )

        self.block5 = nn.Sequential(

            ResidualBlock(256,512),

            ResidualBlock(512,512,2)

        )

        

        self.avgpool = nn.AvgPool2d(2)

        # vowel_diacritic

        self.fc1 = nn.Linear(512,11)

        # grapheme_root

        self.fc2 = nn.Linear(512,168)

        # consonant_diacritic

        self.fc3 = nn.Linear(512,7)

        

    def forward(self,x):

        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        x = self.block4(x)

        x = self.block5(x)

        x = self.avgpool(x)

        x = x.view(x.size(0),-1)

        x1 = self.fc1(x)

        x2 = self.fc2(x)

        x3 = self.fc3(x)

        return x1,x2,x3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ResNet18().to(device)

model.load_state_dict(torch.load('/kaggle/input/grapheme-resnet-18-naive-learning-3/saved_weights.pth'))
def Resize(df,size=64):

    resized = {} 

    df = df.set_index('image_id')

    for i in tqdm(range(df.shape[0])):

        image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size))

        resized[df.index[i]] = image.reshape(-1)

    resized = pd.DataFrame(resized).T.reset_index()

    resized.columns = resized.columns.astype(str)

    resized.rename(columns={'index':'image_id'},inplace=True)

    return resized
model.eval()

test_data = ['test_image_data_0.parquet','test_image_data_1.parquet','test_image_data_2.parquet','test_image_data_3.parquet']

predictions = []

batch_size=1

for fname in test_data:

    data = pd.read_parquet(f'/kaggle/input/bengaliai-cv19/{fname}')

    data = Resize(data)

    test_image = GraphemeDataset(data)

    test_loader = torch.utils.data.DataLoader(test_image,batch_size=1,shuffle=False)

    with torch.no_grad():

        for idx, (inputs) in tqdm(enumerate(test_loader),total=len(test_loader)):

            inputs.to(device)

            

            outputs1,outputs2,outputs3 = model(inputs.unsqueeze(1).float().cuda())

            predictions.append(outputs3.argmax(1).cpu().detach().numpy())

            predictions.append(outputs2.argmax(1).cpu().detach().numpy())

            predictions.append(outputs1.argmax(1).cpu().detach().numpy())

    
submission = pd.read_csv('/kaggle/input/bengaliai-cv19/sample_submission.csv')
submission.target = np.hstack(predictions)
submission.head(10)
submission.to_csv('submission.csv',index=False)