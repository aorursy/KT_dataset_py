!ls
!ls ../input
!ls ../input/train/ | head -5
!head -n 2 ../input/train.csv
from __future__ import print_function, division

import os

import torch

import pandas as pd

from skimage import io, transform

import numpy as np

import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, utils



# Ignore warnings

import warnings

warnings.filterwarnings("ignore")

train_data = pd.read_csv("../input/train.csv")
train_data.head()
image_name = train_data.iloc[0,0]
plt.title(str(train_data.iloc[0,1]))

plt.imshow(io.imread(os.path.join("../input/train/",image_name)))

#type of the variable that stores id

type(train_data.iloc[0,1])
class WhaleIdDataset(Dataset):

    """the whale identification dataset"""

    def __init__(self, csv, rootDir, transform=None ):

        """

        Args:

            csv (string): path to the file with Id

            rootDir (string): path to the root directory

            transform (callable,optional): optional 

                transforms to be applied on the samples.

        """

        self.data_frame = pd.read_csv(csv)

        self.root_dir = rootDir

        self.transform = transform

        

    def __len__(self):

        return len(self.data_frame)

    

    def __getitem__(self,idx):

        image_name = self.data_frame.iloc[idx,0]

        image_path = os.path.join(self.root_dir,

                                  image_name)

        image = io.imread(image_path)

        Id = self.data_frame.iloc[idx,1]

        sample = {"image": image,"id": Id}

        

        if self.transform:

            sample = self.transform(sample)

        

        return sample 

        
whaleDataset = WhaleIdDataset("../input/train.csv",

                              "../input/train/")
whaleDataset[1]["image"].shape
def show_image(image, id):

    plt.title(id)

    plt.imshow(image)

fig = plt.figure()

for i in range(3):

    a = fig.add_subplot(1, 3, i+1)

    a.axis("off")

    a.set_title(whaleDataset[i]["id"])

    plt.imshow(whaleDataset[i]["image"])