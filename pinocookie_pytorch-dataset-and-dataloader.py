import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
print(torch.__version__)
class DatasetMNIST(Dataset):
    
    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, i don't use ToTensor() method of torchvision.transforms
        # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)
        image = self.data.iloc[index, 1:].values.astype(np.uint8).reshape((1, 28, 28))
        label = self.data.iloc[index, 0]
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label
train_dataset = DatasetMNIST('../input/train.csv', transform=None)
# we can access and get data with index by __getitem__(index)
img, lab = train_dataset.__getitem__(0)
print(img.shape)
print(type(img))
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
train_iter = iter(train_loader)
print(type(train_iter))
images, labels = train_iter.next()

print('images shape on batch size = {}'.format(images.size()))
print('labels shape on batch size = {}'.format(labels.size()))
# make grid takes tensor as arg
# tensor : (batchsize, channels, height, width)
grid = torchvision.utils.make_grid(images)

plt.imshow(grid.numpy().transpose((1, 2, 0)))
plt.axis('off')
plt.title(labels.numpy());
class DatasetMNIST2(Dataset):
    
    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, we use ToTensor(), so we define the numpy array like (H, W, C)
        image = self.data.iloc[index, 1:].values.astype(np.uint8).reshape((28, 28, 1))
        label = self.data.iloc[index, 0]
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label
train_dataset2 = DatasetMNIST2('../input/train.csv', transform=torchvision.transforms.ToTensor())
img, lab = train_dataset2.__getitem__(0)

print('image shape at the first row : {}'.format(img.size()))

train_loader2 = DataLoader(train_dataset2, batch_size=8, shuffle=True)

train_iter2 = iter(train_loader2)
print(type(train_iter2))

images, labels = train_iter2.next()

print('images shape on batch size = {}'.format(images.size()))
print('labels shape on batch size = {}'.format(labels.size()))
grid = torchvision.utils.make_grid(images)

plt.imshow(grid.numpy().transpose((1, 2, 0)))
plt.axis('off')
plt.title(labels.numpy());
transform = transforms.Compose([
    transforms.ToPILImage(), # because the input dtype is numpy.ndarray
    transforms.RandomHorizontalFlip(0.5), # because this method is used for PIL Image dtype
    transforms.ToTensor(), # because inpus dtype is PIL Image
])
train_dataset3 = DatasetMNIST2('../input/train.csv', transform=transform)
train_loader3 = DataLoader(train_dataset3, batch_size=8, shuffle=True)

train_iter3 = iter(train_loader3)
print(type(train_iter3))

images, labels = train_iter3.next()

print('images shape on batch size = {}'.format(images.size()))
print('labels shape on batch size = {}'.format(labels.size()))
grid = torchvision.utils.make_grid(images)

plt.imshow(grid.numpy().transpose((1, 2, 0)))
plt.axis('off')
plt.title(labels.numpy());

