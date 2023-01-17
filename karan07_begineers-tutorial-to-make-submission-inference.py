# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import torch

import torchvision

from torchvision import transforms, datasets, models

import os

from torch.utils.data import Dataset, DataLoader

import pandas as pd

from PIL import Image

import torch.nn as nn

import numpy as np

from tqdm import tqdm_notebook

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# define batches and input size

num_classes = 5

batch_size = 16

input_size = (x, y)



test_dir = '../input/'
# data transforms

data_transforms = transforms.Compose([

        transforms.Resize(input_size),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
# class return images in 'test.csv'

class ImageDataset(Dataset):

    

    def __init__(self, csv_file, root_dir, transform = None):

        self.class_labels = pd.read_csv(csv_file)

        self.root_dir = root_dir

        self.transform = transform

        

    def __len__(self):

        return len(self.class_labels)

    

    def __getitem__(self, idx):

        image_name = os.path.join(self.root_dir, self.class_labels.iloc[idx, 0])

        image = Image.open(image_name + '.png')



        sample = {'image': image}

        

        if self.transform:

            sample['image'] = self.transform(sample['image'])

            

        return sample
# create test dataset and loader

test_dataset = ImageDataset(test_dir + 'test.csv', test_dir +'/test_images/', data_transforms)

test_loader = DataLoader(test_dataset, batch_size, shuffle=False,  num_workers=6)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
len(test_dataset)
# load model here

model_ft = 'load_model_here'

num_ftrs = model_ft.fc.in_features

model_ft.fc = nn.Linear(num_ftrs, num_classes)

model_ft = model_ft.to(device)



# load weights

state = torch.load('*.pth')

model_ft.load_state_dict(state['state_dict'])
# predictions

pred_list = []

model_ft.eval()

for sample in test_loader:

    image = sample['image'].to(device)

    outputs = model_ft(image)

    _, predicted = torch.max(outputs, 1) 

    pred_list += [p.item() for p in predicted]
# make submission to csv

submission = pd.read_csv(test_dir+'test.csv')

submission['diagnosis'] = pred_list

submission.to_csv(test_dir + 'submission.csv', index=False)