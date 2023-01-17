#import custom_models

#python packages
from PIL import Image
from tqdm.notebook import tqdm
#from tqdm import tqdm
import gc
import datetime
import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from skimage import io
#torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
#torchvision
import torchvision
from torchvision import datasets, models, transforms
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using GPU!")
else:
    print("WARNING: Could not find GPU! Using CPU only")
class MultimodalDataset(Dataset):
    """
    Custom dataset definition
    """
    def __init__(self, csv_path, img_path, transform=None):
        """
        """
        self.df = pd.read_csv(csv_path)
        self.img_path = img_path
        self.transform = transform
        
            
    def __getitem__(self, index):
        """
        """
        img_name = self.df.iloc[index]["image_name"] 
        img_path = os.path.join(self.img_path, img_name)
        image = Image.open(img_path)
        image = image.convert("RGB")
        image = np.asarray(image)
        if self.transform is not None:
            image = self.transform(image)
            
        dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor # ???
        features = np.fromstring(self.df.iloc[index]["features"][1:-1], sep=",") #turns features into an array
        features = torch.from_numpy(features.astype("float")) #turns the features array into a vector
        #label = int(self.df.iloc[index]['label'])
        labels = torch.tensor(list(self.df.iloc[index]["target"]), dtype = torch.float64)
        #print("Label type: ", type(label))
        #label = np.int_(label) #???
        #print("label type post casting: ", type(label))
        return image, features, labels
        
    def __len__(self):
        return len(self.df)
def get_dataloaders(input_size, batch_size, augment=False, shuffle = True):
    # How to transform the image when you are loading them.
    # you'll likely want to mess with the transforms on the training set.
    
    # For now, we resize/crop the image to the correct input size for our network,
    # then convert it to a [C,H,W] tensor, then normalize it to values with a given mean/stdev. These normalization constants
    # are derived from aggregating lots of data and happen to produce better results.
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            #Add extra transformations for data augmentation
            transforms.RandomApply([
                transforms.RandomChoice([
                    transforms.RandomAffine(degrees=20),
                    transforms.RandomAffine(degrees=0,scale=(0.1, 0.15)),
                    transforms.RandomAffine(degrees=0,translate=(0.2,0.2)),
                    #transforms.RandomAffine(degrees=0,shear=0.15),
                    transforms.RandomHorizontalFlip(p=1.0)
                ] if augment else [transforms.RandomAffine(degrees=0)])#else do nothing
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.225])
        ]),
        #'val': transforms.Compose([
            #transforms.ToPILImage(),
            #transforms.Resize(input_size),
            #transforms.CenterCrop(input_size),
            #transforms.ToTensor(),
            #transforms.Normalize([0.5], [0.225])
        #]),
        'test': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.225])
        ])
    }
    # Create training and validation datasets
    data_subsets = {x: MultimodalDataset(csv_path="../input/melanoma/features_"+x+".csv", 
                                         img_path="../input/siim-isic-melanoma-classification/jpeg/"+x, 
                                         transform=data_transforms[x]) for x in data_transforms.keys()}
    # Create training and validation dataloaders
    # Never shuffle the test set
    dataloaders_dict = {x: DataLoader(data_subsets[x], batch_size=batch_size, shuffle=False if x != 'train' else shuffle, num_workers=4) for x in data_transforms.keys()}
    return dataloaders_dict
shuffle_datasets = True
dataloaders = get_dataloaders(224, 64, shuffle_datasets)
dataloaders