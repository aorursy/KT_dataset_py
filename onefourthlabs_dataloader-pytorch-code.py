import os

from PIL import Image

import matplotlib.pyplot as plt



import torch

import torchvision

from torch.utils.data import DataLoader, Dataset, random_split

import torchvision.transforms as transforms



#For converting the dataset to torchvision dataset format

class VowelConsonantDataset(Dataset):

    def __init__(self, file_path,train=True,transform=None):

        self.transform = transform

        self.file_path=file_path

        self.train=train

        self.file_names=[file for _,_,files in os.walk(self.file_path) for file in files]

        self.len = len(self.file_names)

        if self.train:

            self.classes_mapping=self.get_classes()

    def __len__(self):

        return len(self.file_names)

    

    def __getitem__(self, index):

        file_name=self.file_names[index]

        image_data=self.pil_loader(self.file_path+"/"+file_name)

        if self.transform:

            image_data = self.transform(image_data)

        if self.train:

            file_name_splitted=file_name.split("_")

            Y1 = self.classes_mapping[file_name_splitted[0]]

            Y2 = self.classes_mapping[file_name_splitted[1]]

            z1,z2=torch.zeros(10),torch.zeros(10)

            z1[Y1-10],z2[Y2]=1,1

            label=torch.stack([z1,z2])



            return image_data, label



        else:

            return image_data, file_name

          

    def pil_loader(self,path):

        with open(path, 'rb') as f:

            img = Image.open(f)

            return img.convert('RGB')



      

    def get_classes(self):

        classes=[]

        for name in self.file_names:

            name_splitted=name.split("_")

            classes.extend([name_splitted[0],name_splitted[1]])

        classes=list(set(classes))

        classes_mapping={}

        for i,cl in enumerate(sorted(classes)):

            classes_mapping[cl]=i

        return classes_mapping

    
import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim



import torchvision

import matplotlib.pyplot as plt

from torchvision import datasets



import torchvision.transforms as transforms



import numpy as np

import pandas as pd



train_on_gpu = torch.cuda.is_available()
transform = transforms.Compose([

    transforms.ToTensor()])
full_data = VowelConsonantDataset("../input/padhai-hindi-vow-cons-classification/train/train",train=True,transform=transform)

train_size = int(0.9 * len(full_data))

test_size = len(full_data) - train_size



train_data, validation_data = random_split(full_data, [train_size, test_size])



train_loader = torch.utils.data.DataLoader(train_data, batch_size=60, shuffle=True)

validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=60, shuffle=True)
test_data = VowelConsonantDataset("../input/padhai-hindi-vow-cons-classification/test/test",train=False,transform=transform)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=60,shuffle=False)