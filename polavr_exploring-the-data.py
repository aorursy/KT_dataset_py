import os #to communicate with the opertive system

import numpy as np #vector operations

from PIL import Image



#Libraries related with PyTorch

import torch #PyTorch library

from torch.utils.data import Dataset, random_split, DataLoader

from torchvision.datasets import ImageFolder

import torchvision.transforms as tt



#Useful libraries for plotting

from torchvision.utils import make_grid

import matplotlib.pyplot as plt

%matplotlib inline  

#plots in the line below the code, inside the notebook 
train_dir = '/kaggle/input/twoclass-weather-classification/train' 

classes = os.listdir(train_dir)

#num_classes = len(classes)

print(classes)
test_dir = '/kaggle/input/twoclass-weather-classification/test' 

classes = os.listdir(train_dir)

#num_classes = len(classes)

print(classes)
#Train dataset



sunny_dir = '/kaggle/input/twoclass-weather-classification/train/sunny/'

sunny_files = os.listdir(sunny_dir)

print(len(sunny_files))



cloudy_dir = '/kaggle/input/twoclass-weather-classification/train/cloudy/'

cloudy_files = os.listdir(cloudy_dir)

print(len(cloudy_files))
#Test dataset



test_sunnydir = '/kaggle/input/twoclass-weather-classification/test/sunny/'

test_sunny_files = os.listdir(test_sunnydir)

print(len(test_sunny_files))



test_cloudydir = '/kaggle/input/twoclass-weather-classification/test/cloudy/'

test_cloudy_files = os.listdir(test_cloudydir)

print(len(test_cloudy_files))
print('Examples of the train sunny files: ' + str(sunny_files[:5]))

print('Examples of the train cloudy files: ' + str(cloudy_files[:5]))
#Using PIL library and pyplot

plt.imshow(Image.open(sunny_dir+sunny_files[0]))
plt.imshow(Image.open(cloudy_dir+cloudy_files[2974]))
from torchvision.datasets import ImageFolder

dataset = ImageFolder(train_dir, transform=tt.ToTensor())

test_dataset = ImageFolder(test_dir, transform=tt.ToTensor())
#Using the matplotlib.pyplot (plt) library

def show_example(img, label):

    print('Label: ', dataset.classes[label], "("+str(label)+")")

    plt.imshow(img.permute(1, 2, 0))
img, label = dataset[0]

print(img.shape, label)

img
show_example(*dataset[0])
img, label = dataset[5000]

print(img.shape, label)

img
show_example(*dataset[5000])
img, label = test_dataset[0]

print(img.shape, label)

img
show_example(*test_dataset[0])
img, label = test_dataset[100]

print(img.shape, label)

img
show_example(*test_dataset[100])
batch_size = 64
train_loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)

test_loader = DataLoader(test_dataset, batch_size, num_workers=4, pin_memory=True)
def show_batch(dl):

    for images, labels in dl:

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.set_xticks([]); ax.set_yticks([])

        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0)) 

        #Permutation 

        break
show_batch(train_loader)
show_batch(test_loader)
#Import Jovian to commit and save my work there

!pip install jovian --upgrade -q

import jovian
projectName = 'Exploring data_Binary weather classification'

jovian.commit(project=projectName, environment=None)