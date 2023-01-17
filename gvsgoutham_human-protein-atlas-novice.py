# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install jovian
import os

import torch

import pandas as pd

import numpy as np

from torch.utils.data import Dataset, random_split, DataLoader

from PIL import Image

import torchvision.models as models

import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

import torchvision.transforms as T

import sklearn.metrics as m

import torch.nn.functional as F

import torch.nn as nn

from torchvision.utils import make_grid

from sklearn.model_selection import train_test_split

import jovian

from torchvision.datasets import ImageFolder



%matplotlib inline
project_name = 'Human Protien Atlas - Novice'
directory = '../input/jovian-pytorch-z2g/Human protein atlas'



csv_df = pd.read_csv(directory + '/train.csv')



labels = {0: 'Mitochondria',

1: 'Nuclear bodies',

2: 'Nucleoli',

3: 'Golgi apparatus',

4: 'Nucleoplasm',

5: 'Nucleoli fibrillar center',

6: 'Cytosol',

7: 'Plasma membrane',

8: 'Centrosome',

9: 'Nuclear speckles'}
print(labels)

csv_df.head()
train_trans = T.Compose([

#     T.RandomCrop(512, padding=8, padding_mode='reflect'),

#     T.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1, 1)), 

#     T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),

#     T.RandomHorizontalFlip(), 

#     T.RandomRotation(10),

    T.ToTensor(), 

#     T.Normalize(*imagenet_stats,inplace=True), 

#     T.RandomErasing(inplace=True)

])



valid_trans = T.ToTensor()

    
class HumanProteinDataset(Dataset):

    def __init__(self, df, root_dir, transform=None):

        self.df = df

        self.transform = transform

        self.root_dir = root_dir

        

    def __len__(self):

        return len(self.df)    

    

    def __getitem__(self, idx):

        row = self.df.loc[idx]

        img_id, img_label = row['Image'], row['Label']

        img_fname = self.root_dir + "/" + str(img_id) + ".png"

        img = Image.open(img_fname)

        if self.transform:

            img = self.transform(img)

        return img,encode_label(img_label), decode_target(encode_label(img_label),text_labels = True)

def encode_label(label):

    target = torch.zeros(10)

    for l in str(label).split(' '):

        target[int(l)] = 1.

    return target



def decode_target(target, text_labels=False, threshold=0.5):

    result = []

    for i, x in enumerate(target):

        if (x >= threshold):

            if text_labels:

                result.append(labels[i] + "(" + str(i) + ")")

            else:

                result.append(str(i))

    return ' '.join(result)
np.random.seed(30)

indices = np.random.randn(len(csv_df)) < 0.8

train_df,val_df = csv_df[indices].reset_index(),csv_df[~indices].reset_index()

print(train_df.head())

print(val_df.head())
train_ds = HumanProteinDataset(train_df,directory + '/train/',transform = train_trans)

val_ds = HumanProteinDataset(val_df,directory + '/train/',transform = valid_trans)
plt.imshow(train_ds[0][0].permute(1,2,0))

print(train_ds[0][2])
batch_size = 30

train_dl = DataLoader(train_ds,batch_size,shuffle = True,num_workers = 4,pin_memory = True)

val_dl = DataLoader(val_ds,batch_size*3,num_workers = 4,pin_memory = True)
for i,_,l in train_dl:

    fig,ax = plt.subplots(figsize = (20,10))

    ax.imshow(make_grid(i,nrow = 10).permute(1,2,0))

    print(l)

    break