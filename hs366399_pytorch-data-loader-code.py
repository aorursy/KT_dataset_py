#importing required libraries

import os

import torch

import numpy as np

import torchvision.transforms as transforms

import torchvision.datasets as datasets

from PIL import Image

from torch.utils.data import DataLoader, Dataset, random_split



device = torch.device('cuda:0')

data_path = {'train' : "../input/train/train/", 'test' : "../input/test/test/"}

print(torch.cuda.get_device_name(0))
#For converting the dataset to torchvision dataset format

class HindiVowelConsonantDataset(Dataset):

    

    def __init__(self, data_path, transform = None, train = True):

        self.train_img_path = data_path['train']

        self.test_img_path = data_path['test']

        self.train_img_files = os.listdir(self.train_img_path)

        self.test_img_files = os.listdir(self.test_img_path)

        self.transform = transform

        self.train = train

    

    def __len__(self):

        return len(self.train_img_files)

    

    def __getitem__(self, indx):

            

        if self.train:  

            

            if indx >= len(self.train_img_files):

                raise Exception("Index should be less than {}".format(len(self.train_img_files)))

               

            image = Image.open(self.train_img_path + self.train_img_files[indx]).convert('RGB')

            labels = self.train_img_files[indx].split('_')

            V = int(labels[0][1])

            C = int(labels[1][1])

            label = {'Vowel' : V, 'Consonant' : C}



            if self.transform:

                image = self.transform(image)



            return image, label

        

        if self.train == False:

            image = Image.open(self.test_img_path + self.test_img_files[indx]).convert('RGB')

            if self.transform:

                image = self.transform(image)



            return image, self.test_img_files[indx]
transform = transforms.Compose([transforms.ToTensor(),

                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
data = HindiVowelConsonantDataset(data_path, transform = transform, train = True)



train_size = int(0.9 * len(data))

test_size = len(data) - train_size



train_data, validation_data = random_split(data, [train_size, test_size])



train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=32, shuffle=False)
test_data = HindiVowelConsonantDataset(data_path, transform = transform, train = False)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=32,shuffle=False)