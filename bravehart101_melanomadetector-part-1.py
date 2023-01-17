import os

import torch

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



from PIL import Image



from torch.utils.data import DataLoader, Dataset, random_split

import torchvision.models as models

import torchvision.transforms as transforms

import torch.nn.functional as F

import torch.nn as nn

from torchvision.utils import make_grid
train_df = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')

train_df.head()
train_df.drop(columns = ['patient_id'], axis = 1, inplace = True)

train_df.head()
null_df = train_df.isnull().sum().to_frame()

null_df.columns = ['null_vals']

null_df['percent_null'] = null_df['null_vals']/len(train_df)

null_df
EDA_df = train_df.dropna()

len(EDA_df), len(train_df)
sns.set_style('darkgrid')



f, ax = plt.subplots(1, 2, figsize = (16, 8))

sns.countplot(y = 'diagnosis', data = EDA_df, ax = ax[0])

sns.countplot(x = 'target', data = EDA_df, ax = ax[1])



f.show()



print('Percentage of different diagnosed mole types:')

EDA_df['diagnosis'].value_counts() / len(EDA_df) * 100
target_grouped = EDA_df.groupby('target')

confirmed_df = target_grouped.get_group(1)

confirmed_df
f, ax = plt.subplots(1,2,figsize = (12, 6))

sns.countplot( x = 'sex', data = confirmed_df, ax = ax[0])

sns.violinplot( x = 'sex', y = 'age_approx', data = confirmed_df, inner = 'quartile', ax = ax[1])

f.show()
f, ax = plt.subplots(1, 2, figsize = (20, 8))

sns.countplot( x = 'anatom_site_general_challenge', data = confirmed_df, ax = ax[0])

sns.countplot( x = 'anatom_site_general_challenge', hue = 'sex', data = confirmed_df, ax = ax[1])

f.show()
sns.set_style('white')



def train_img_viewer(index):

    """Shows image in the training dataset at the random index that the user provides.

    

    Args-

        index- index of the image

    Returns-

        None

    """

    

    img_info = train_df.loc[index,['image_name', 'benign_malignant']]

    img_name = img_info.image_name + '.jpg'

    img_class = img_info.benign_malignant

    path = '../input/siim-isic-melanoma-classification/jpeg/train'

    img_path = os.path.join(path, img_name)

    img = Image.open(img_path)

    transform = transforms.ToTensor()

    img = transform(img)

    print(img_class)

    plt.imshow(img.permute(1,2,0))
train_img_viewer(1)
train_img_viewer(91)