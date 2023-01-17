import numpy as np
import pandas as pd
import gc
import cv2

import warnings
warnings.filterwarnings('ignore')

import os
import glob
import os.path as osp
from PIL import Image

import matplotlib.pyplot as plt
%matplotlib inline
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torch.utils import data

from torchvision.utils import save_image
from torchvision.datasets import MNIST

import torch
import torch.nn as nn
import torch.nn.functional as F

!ls -all /kaggle/input/dataset-upload-to-kaggle/finished/train/dataraw/hires/
#fungsi imshow image
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
path = '/kaggle/input/dataset-upload-to-kaggle/finished/train/dataraw/hires/'
#class Dataset
class SRDataset(data.Dataset):
    #Fungsi Inialisasi Dataset dari directory menggunkan cv2
    def __init__(self, path, transforms=None, imgloader = cv2.imread):
        self.filenames = []
        self.path = path
        self.transforms = transforms
        filenames = sorted(glob.glob(osp.join(path, '*.jpg')))
        for fn in filenames:
            self.filenames.append(fn)
        self.len = len(self.filenames)
        self.imgloader = imgloader

        
    #fungsi mengambil data dari directory dan  
    def __getitem__(self, index):
        
        #unutk mengambil data darilokal dengan menggunakan parameter imgloader
        #Note apabila pengambilan menggunakan cv2 makan harus diconvert terlebih dahulu dari BGR ke RGB
        target = self.imgloader(self.filenames[index])
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target = cv2.resize(target, (512, 360))
        
        #melakukan load data lalu meresize
        image = cv2.resize(target, (64, 45))
        image = cv2.resize(image, (512, 360))
        
        #statment pendeklarasian variabel image dan target 
        if self.transforms != None:
            image = self.transforms(image)
            target = self.transforms(target)
            
        #pengembalian image dan target (proses perbandingan antara data gambar dengan data target(lowres))
        return (image,target)
    
    #fungsi menghitung panjang
    def __len__(self):
        return self.len
#directory data train dan data valid
train_path = '/kaggle/input/dataset-upload-to-kaggle/finished/train/dataraw/hires/'
valid_path = '/kaggle/input/dataset-upload-to-kaggle/finished/valid/dataraw/hires/'

#penggunaan transforms
tmft = transforms.Compose([
    transforms.ToTensor()
])

#pembukusan data dari directory local menggunakan variabel train dan valid
train_dataset = SRDataset(path=train_path, transforms=tmft)
valid_dataset = SRDataset(path=valid_path, transforms=tmft)

#mengatur settingan dataloader
train_loader = data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
valid_loader = data.DataLoader(valid_dataset, batch_size=16, shuffle=True, num_workers=0)
# get some images
dataiter = iter(train_loader)
image,target = dataiter.next()

imshow(image[0]);plt.show()
imshow(target[0]);plt.show()
