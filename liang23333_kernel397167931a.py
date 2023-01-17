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
pwd
!pip install /kaggle/input/pretrained/pretrainedmodels-0.7.4/pretrainedmodels-0.7.4/
import os

import shutil

if not os.path.exists('/tmp/.cache/torch/checkpoints/'):

    os.makedirs('/tmp/.cache/torch/checkpoints/')

shutil.copy('/kaggle/input/pretrained/densenet121-fbdb23505.pth','/tmp/.cache/torch/checkpoints/')
model_name = 'densenet121'

channels = 1024

loss_weight1 = 0.8

loss_weight2 = 0.1

loss_weight3 = 0.1

import cv2
# -*- coding: utf-8 -*-

import torch

import torch.nn as nn

import torchvision

import torch.nn.functional as F

from torch.nn import Sequential

import pretrainedmodels

import torchvision.models as models



IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]

IMAGE_RGB_STD  = [0.229, 0.224, 0.225]





class LinearBlock(nn.Module):



    def __init__(self, in_features, out_features, bias=True,

                 use_bn=True, activation=F.relu, dropout_ratio=-1, residual=False,):

        super(LinearBlock, self).__init__()

        if in_features is None:

            self.linear = LazyLinear(in_features, out_features, bias=bias)

        else:

            self.linear = nn.Linear(in_features, out_features, bias=bias)

        if use_bn:

            self.bn = nn.BatchNorm1d(out_features)

        if dropout_ratio > 0.:

            self.dropout = nn.Dropout(p=dropout_ratio)

        else:

            self.dropout = None

        self.activation = activation

        self.use_bn = use_bn

        self.dropout_ratio = dropout_ratio

        self.residual = residual

    def __call__(self, x):

        h = self.linear(x)

        if self.use_bn:

            h = self.bn(h)

        if self.activation is not None:

            h = self.activation(h)

        if self.residual:

            h = residual_add(h, x)

        if self.dropout_ratio > 0:

            h = self.dropout(h)

        return h



class RGB(nn.Module):

    def __init__(self,):

        super(RGB, self).__init__()

        self.register_buffer('mean', torch.zeros(1,3,1,1))

        self.register_buffer('std', torch.ones(1,3,1,1))

        self.mean.data = torch.FloatTensor(IMAGE_RGB_MEAN).view(self.mean.shape)

        self.std.data = torch.FloatTensor(IMAGE_RGB_STD).view(self.std.shape)



    def forward(self, x):

        print(self.mean,self.std)

        x = (x-self.mean)/self.std

        return x







class classifier(nn.Module):

    def __init__(self,n_classes=[168,11,7],use_bn=True):

        super(classifier, self).__init__()

        #self.conv0 = nn.Conv2d(1,3,kernel_size=3,stride=1,padding=1)

        self.base_model = pretrainedmodels.__dict__[model_name](pretrained='imagenet')

        #print(*list(self.base_model.children()))

        #exit()

        

        self.backbone = nn.Sequential(*list(self.base_model.children())[:-1]) 



        inch = channels





        self.lin1 = LinearBlock(inch, 512, use_bn=use_bn, activation=F.leaky_relu, dropout_ratio = 0.5, residual=False)

        

        self.head1 = LinearBlock(512, 168, use_bn=False, activation=None, residual=False)



        self.head2 = LinearBlock(512, 11, use_bn=False, activation=None, residual=False)



        self.head3 = LinearBlock(512, 7, use_bn=False, activation=None, residual=False)





    def forward(self,x):

        x_ = x

        feat = self.backbone(x_)

        h = torch.mean(feat, dim=(-1,-2))

        h = self.lin1(h)

        

        x1 = self.head1(h)



        x2 = self.head2(h)



        x3 = self.head3(h)



        return x1,x2,x3



        





model = classifier().cuda()



if os.path.exists('/kaggle/input/densenet/densenet121_86.pth'):

    model.load_state_dict(torch.load('/kaggle/input/densenet/densenet121_86.pth'))

    print("load model success")

else:

    print('load model fail')

model.eval()
from torch.utils.data import Dataset,DataLoader

import pandas as pd

import numpy as np

import cv2

H=137

W=236



class GraphemeDataset(Dataset):#测试dataset

    def __init__(self, fname):

        print(fname)

        self.df = pd.read_parquet(fname)

        self.data = 255 - self.df.iloc[:, 1:].values.reshape(-1, H, W).astype(np.uint8)



    def __len__(self):

        return len(self.data)



    def __getitem__(self, idx):

        name = self.df.iloc[idx,0]

        img = np.reshape(self.data[idx],(H,W))

        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

        img = img.astype(np.float32)/255.0

        img = np.transpose(img,(2,0,1))

        mean = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)

        std = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)

        img = (img-mean)/std

        img = torch.FloatTensor(img)

        return img, name



test_data = ['/kaggle/input/bengaliai-cv19/test_image_data_0.parquet','/kaggle/input/bengaliai-cv19/test_image_data_1.parquet','/kaggle/input/bengaliai-cv19/test_image_data_2.parquet',

             '/kaggle/input/bengaliai-cv19/test_image_data_3.parquet']
from tqdm import tqdm_notebook as tqdm

row_id,target = [],[]

for fname in test_data:

    #data = pd.read_parquet(f'/kaggle/input/bengaliai-cv19/{fname}')

    test_image = GraphemeDataset(fname)

    dl = torch.utils.data.DataLoader(test_image,batch_size=128,num_workers=12,shuffle=False)

    with torch.no_grad():

        for x, y in tqdm(dl):

            x = x.float().cuda()

            p1,p2,p3 = model(x)

            p1 = p1.argmax(-1).view(-1).cpu()

            p2 = p2.argmax(-1).view(-1).cpu()

            p3 = p3.argmax(-1).view(-1).cpu()

            for idx,name in enumerate(y):

                row_id += [f'{name}_grapheme_root',f'{name}_vowel_diacritic',

                           f'{name}_consonant_diacritic']

                target += [p1[idx].item(),p2[idx].item(),p3[idx].item()]

                

sub_df = pd.DataFrame({'row_id': row_id, 'target': target})

sub_df.to_csv('submission.csv', index=False)

sub_df.head(20)