import numpy as np

import pandas as pd

import matplotlib.pyplot as  plt

import matplotlib.image as mpimg

import seaborn as sns

import os

from sklearn.model_selection import train_test_split

import torch

import torch.nn as nn

import torch.nn.functional as F

from torchvision.datasets import ImageFolder

from torch.utils.data import Subset

from torch.utils.data import DataLoader, Dataset

from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, RandomAffine, Grayscale, Resize

from torch.utils.data.sampler import SubsetRandomSampler

import torch.optim as optim

import skimage.io as io

from glob import glob

from PIL.Image import fromarray
xray_data = pd.read_csv('../input/data/Data_Entry_2017.csv')



my_glob = glob('../input/data/images*/images/*.png')



full_img_paths = {os.path.basename(x): x for x in my_glob}

xray_data['full_path'] = xray_data['Image Index'].map(full_img_paths.get)



dummy_labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening', 

'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'] # taken from paper



for label in dummy_labels:

    xray_data[label] = xray_data['Finding Labels'].map(

        lambda result: 1.0 if label in result else 0

    )

    

xray_data['target_vector'] = xray_data.apply(

    lambda target: [target[dummy_labels].values], 1).map(

    lambda target: target[0]

)
class XRayDatasetFromCSV(Dataset):

    def __init__(self, csv_file, transform=None):

        self.data = csv_file

        self.labels = np.stack(self.data['target_vector'].values)

        self.transform = transform

    

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, index):

        

        img_name = self.data['full_path'][index]

        image = io.imread(img_name)

        

        if self.transform:

            image = self.transform(fromarray(image))

            

        return image, self.data['target_vector'].values[index]
image_size = (128, 128)



transform = Compose([

    Resize(image_size),

    Grayscale(),

    RandomHorizontalFlip(),

    RandomAffine(degrees=20, shear=(-0.2, 0.2, -0.2, 0.2), scale=(0.8, 1.2)),

    ToTensor()

])
dataset = XRayDatasetFromCSV(xray_data, transform)
test_split = 0.2

indices = list(range(len(xray_data)))

split = int(np.floor(test_split * len(xray_data)))



train_indices, test_indices = indices[split:], indices[:split]



train_sampler = SubsetRandomSampler(train_indices)

test_sampler = SubsetRandomSampler(test_indices)
train_loader = DataLoader(dataset=dataset, batch_size=32, sampler=train_sampler)

test_loader = DataLoader(dataset=dataset, batch_size=128, sampler=test_sampler)
a = 1

global image_to_print

try:

    for batch_idx, image in enumerate(dataset):

        a += 1

        image_to_print = image

except:

    print(image_to_print)

    

    

try:

    for batch_idx, (image, target) in enumerate(train_loader):

        a += 1

        image_to_print = image

except:

    print(image_to_print)

    

    

try:

    for batch_idx, (image, target) in enumerate(test_loader):

        a += 1

        image_to_print = image

except:

    print(image_to_print)
class Flatten(nn.Module):

    def forward(self, x):

        return x.view(x.shape[0], -1)
class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.seq = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            nn.Dropout(p=0.2),

            

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            nn.Dropout(p=0.2),

            

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            nn.Dropout(p=0.2),

            

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            nn.Dropout(p=0.2),

            

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            nn.Dropout(p=0.2),

            

            Flatten(),

            

            nn.Linear(128 * 4 * 4, 500),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(500, len(dummy_labels)),

            nn.Softmax()

        )

        

    def forward(self, x):

        return self.seq(x)
net = Net()
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters())