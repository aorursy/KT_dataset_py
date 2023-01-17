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

import numpy as np

import torch.nn as nn

from torch.utils.data import Dataset,DataLoader

from torchvision import transforms,models

#from transform import get_train_transform

import pandas as pd

import torch.nn.functional as F



# 判定GPU是否存在

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#GPU是否存在

# device = torch.device("cuda:0")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# device = 'cuda:0'

print ('device:', device)



INPUT_PATH = "/kaggle/input/bengaliai-cv19/"

CKPT_path = '/kaggle/input/' + '/ckpt11/resnet50_enc_4.ckpt'



HEIGHT = 137

WIDTH = 236

TARGET_SIZE = 256

# 定义超参数

batch_size = 64

SIZE = 128



# 中心填充

def make_square(img, target_size=256):

    img = img[0:-1, :]

    height, width = img.shape



    x = target_size

    y = target_size



    square = np.ones((x, y), np.uint8) * 255

    square[(y - height) // 2:y - (y - height) // 2, (x - width) // 2:x - (x - width) // 2] = img



    return square





# 模型 

import cv2

import torch

import torch.nn as nn

from torch.utils.data import Dataset,DataLoader

from torchvision import transforms,models





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

            nn.Conv2d(3,64,kernel_size=2,stride=2,padding=3,bias=False),

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

        self.fc1 = nn.Linear(8192,11)

        # grapheme_root

        self.fc2 = nn.Linear(8192,168)

        # consonant_diacritic

        self.fc3 = nn.Linear(8192,7)

        

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

# 等价于删除某个层

class Identity(torch.nn.Module):

    def __init__(self):

        super(Identity, self).__init__()

        

    def forward(self, x):

        return x





def resnet50():

    res50 = models.resnet50(pretrained=False)

    res50.fc = Identity()  # 最后一层全连接删除

    res50.avgpool = Identity() # 全局平均池化也需要更改，原始的224降低32x，现在32降低32x成了1

    return res50

    

    

class ResNet50(torch.nn.Module):

    def __init__(self,):

        super(ResNet50, self).__init__()

        self.resnet50 = resnet50()

        self.avgpool = nn.AvgPool2d(2)

        # vowel_diacritic

        self.fc1 = nn.Linear(32768,11)

        # grapheme_root

        self.fc2 = nn.Linear(32768,168)

        # consonant_diacritic

        self.fc3 = nn.Linear(32768,7)

        

    def forward(self, x):

        x = self.resnet50(x)

        # x = self.avgpool(x)

        x = x.view(x.size(0),-1)

        x1 = self.fc1(x)

        x2 = self.fc2(x)

        x3 = self.fc3(x)

        return x1,x2,x3



# 数据增强

#!usr/bin/env python  

#-*- coding:utf-8 _*- 

import random

import math

import torch



from PIL import Image, ImageOps, ImageFilter

from torchvision import transforms





def update_lr(optimizer, lr):    

    for param_group in optimizer.param_groups:

        param_group['lr'] = lr



class Resize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):

        self.size = size

        self.interpolation = interpolation



    def __call__(self, img):

        # padding

        ratio = self.size[0] / self.size[1]

        w, h = img.size

        if w / h < ratio:

            t = int(h * ratio)

            w_padding = (t - w) // 2

            img = img.crop((-w_padding, 0, w+w_padding, h))

        else:

            t = int(w / ratio)

            h_padding = (t - h) // 2

            img = img.crop((0, -h_padding, w, h+h_padding))



        img = img.resize(self.size, self.interpolation)



        return img



class RandomRotate(object):

    def __init__(self, degree, p=0.5):

        self.degree = degree

        self.p = p



    def __call__(self, img):

        if random.random() < self.p:

            rotate_degree = random.uniform(-1*self.degree, self.degree)

            img = img.rotate(rotate_degree, Image.BILINEAR)

        return img



class RandomGaussianBlur(object):

    def __init__(self, p=0.5):

        self.p = p

    def __call__(self, img):

        if random.random() < self.p:

            img = img.filter(ImageFilter.GaussianBlur(

                radius=random.random()))

        return img





def get_train_transform(size):

    train_transform = transforms.Compose([

        #Resize((int(size * (256 / 224)), int(size * (256 / 224)))),

        transforms.RandomCrop(size),

        #transforms.RandomHorizontalFlip(),

        RandomRotate(15, 0.3),

        RandomGaussianBlur(),

        transforms.ToTensor(),

        # transforms.Normalize(mean=mean, std=std),

    ])

    return train_transform



def get_test_transform(size):

    return transforms.Compose([

        #Resize((int(size * (256 / 224)), int(size * (256 / 224)))),

        transforms.CenterCrop(size),

        transforms.ToTensor(),

        # transforms.Normalize(mean=mean, std=std),

    ])





from PIL import Image

import cv2

# 中心填充

def default_loader(path):      

    # 注意要保证每个batch的tensor大小时候一样的。      

    return Image.open(path).convert('RGB')

    #return cv2.imread(path)





# 数据加载模块

class BengaliParquetDatasetTest(Dataset):

    def __init__(self, parquet_file, transform=None, _type="train"):

        self.data = pd.read_parquet(parquet_file)

        self.transform = transform

        self.type = _type



    def __len__(self):

        return len(self.data)



    def __getitem__(self, idx):

        if self.type == "train":

            return None

        

        if self.type == "test":

            tmp = self.data.iloc[idx, 1:].values.reshape(HEIGHT, WIDTH)

            img = np.zeros((TARGET_SIZE, TARGET_SIZE, 3))

            img[..., 0] = make_square(tmp, target_size=TARGET_SIZE)

            img[..., 1] = img[..., 0]

            img[..., 2] = img[..., 0]

            image_id = self.data.iloc[idx, 0]

            print ('image_id:', image_id)

            if self.transform:

                img = Image.fromarray(img.astype('uint8')).resize((130, 130))

                img = self.transform(img)

            return image_id, img





submission_df = pd.read_csv(INPUT_PATH + '/sample_submission.csv')

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = ResNet50().to(device)

# model.load_state_dict(torch.load( CKPT_path, map_location=torch.device('cuda:0') ))

model.load_state_dict(torch.load( CKPT_path, map_location=torch.device('cpu') ))

model.to(device)



transform_test = get_test_transform( SIZE )





results = []

results_id = []

for i in range(4):

    parquet_file = INPUT_PATH + '/test_image_data_{}.parquet'.format(i)

    print ("parq:", parquet_file)

    test_dataset = BengaliParquetDatasetTest(parquet_file=parquet_file, transform=transform_test, _type="test")

    data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,

                                                   num_workers=12, shuffle=False)

    



    print('Parquet {}'.format(i))

    model.eval()

    tk0 = data_loader_test



    for step, (image_id, images) in enumerate(tk0):

        inputs = images

        image_ids = image_id

        inputs = inputs.to(device, dtype=torch.float)



        out_vowel, out_graph, out_conso = model(inputs)

        out_vowel = F.softmax(out_vowel, dim=1).data.cpu().numpy().argmax(axis=1)

        out_graph = F.softmax(out_graph, dim=1).data.cpu().numpy().argmax(axis=1)

        out_conso = F.softmax(out_conso, dim=1).data.cpu().numpy().argmax(axis=1)



        for idx, image_id in enumerate(image_ids):

            results.append(out_conso[idx])

            results.append(out_graph[idx])

            results.append(out_vowel[idx])

            

            results_id.append( str(image_id) + '_consonant_diacritic' ) 

            results_id.append( str(image_id) + '_grapheme_root' )

            results_id.append( str(image_id) + '_vowel_diacritic' )



submission_df['target'] = results

submission_df['row_id'] = submission_df['row_id']

submission_df.to_csv('submission.csv', index=False)



print ('end!!!!!!!!!')



submission_df