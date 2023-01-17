import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('seaborn-whitegrid')



import os



import torch

import torch.nn as nn

from torchvision import datasets, transforms 

from torch.utils.data import Dataset



import cv2
train_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'

test_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/test/'



train = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')

test = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')

sample_submission = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')
class MelanomaDataset(Dataset):

    

    def __init__(self, dataframe, transform=None, test=False):

        self.df = dataframe

        self.transform = transform

        self.test = test

    

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, idx):



        

        img = self.df.image_name.values[idx]

        

        if self.test == False:

            p_path = train_path + img + '.jpg'

            

        else:

            p_path = test_path + img + '.jpg'

            

            

        image = cv2.imread(p_path)

        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        image = transforms.ToPILImage()(image)

        

        if self.transform:

            image = self.transform(image)

            

        if self.test == False:

            label = self.df.target.values[idx]

            

            return image, label

        

        else:

            

            return image
train.head()
train.info()
test.head()
train.benign_malignant.value_counts()
sample_benign = train[train.target == 0][:5]

sample_malig = train[train.target == 1][:5]



sample = pd.merge(sample_benign, sample_malig, how = 'outer')



xforms = transforms.Compose([transforms.ToTensor()])

sampleset = MelanomaDataset(sample, transform= xforms)
fig,ax = plt.subplots(2,5, figsize = (16, 5))

n = 0

for x in range(2):

    for y in range(5):

        ax[x, y].imshow(sampleset[x+y+n][0].permute(1,2,0))

        label = ['malignant' if sampleset[x+y+n][1]==1 else 'benign']

        ax[x, y].set_title(label[0])

    n = 4
diag = train.diagnosis.unique().tolist()

#diag = diag[1:]

diag
parts = train.anatom_site_general_challenge.unique().tolist()

parts = [x for x in parts if str(x) != 'nan']

parts
fig, ax = plt.subplots(1,2, figsize = (16, 6))



ax1 = sns.countplot(x="anatom_site_general_challenge",

                    data=train[train.benign_malignant == 'benign'],

                    order = parts, ax = ax[0])

ax1.set_xticklabels(ax1.get_xticklabels(),

                    rotation=30,

                    horizontalalignment='right')

ax1.set_title('Benign', fontsize = 20, weight = 'bold')

for p in ax1.patches:

    ax1.annotate(f'{p.get_height()}', (p.get_x()+0.17, p.get_height()+100.0))

    

    

ax2 = sns.countplot(x="anatom_site_general_challenge",

                    data=train[train.benign_malignant == 'malignant'],

                    order = parts, ax = ax[1])

ax2.set_xticklabels(ax2.get_xticklabels(),

                    rotation=30,

                    horizontalalignment='right')

ax2.set_title('Malignant', fontsize = 20, weight = 'bold')

for p in ax2.patches:

    ax2.annotate(f'{p.get_height()}', (p.get_x()+0.25, p.get_height()+1.5))    
fig, ax = plt.subplots(1,2, figsize = (16, 6))



ax1 = sns.countplot(x="anatom_site_general_challenge",

                    data=train,

                    order = parts, ax = ax[0])

ax1.set_xticklabels(ax1.get_xticklabels(),

                    rotation=30,

                    horizontalalignment='right')

ax1.set_title('Train', fontsize = 20, weight = 'bold')

for p in ax1.patches:

    ax1.annotate(f'{p.get_height()}', (p.get_x()+0.17, p.get_height()+100.0))

    

    

ax2 = sns.countplot(x="anatom_site_general_challenge",

                    data=test,

                    order = parts, ax = ax[1])

ax2.set_xticklabels(ax2.get_xticklabels(),

                    rotation=30,

                    horizontalalignment='right')

ax2.set_title('Test', fontsize = 20, weight = 'bold')

for p in ax2.patches:

    ax2.annotate(f'{p.get_height()}', (p.get_x()+0.25, p.get_height()+50.5))    
def get_age_df(data):

    

    ages_benign = data[data.target == 0].age_approx.dropna()

    ages_malig = data[data.target == 1].age_approx.dropna()



    ages = pd.DataFrame({'malignant':ages_malig.value_counts(),

                         'benign':ages_benign.value_counts(),

                         }).sort_index(ascending=True)

    return ages
train_ages = get_age_df(train)



train_ages.plot(kind='bar',secondary_y = 'malignant',  figsize = (12, 6))

plt.title("Age and Melanoma : Train", fontsize = 20, weight= 'bold')
fig, ax = plt.subplots(1,2, figsize = (16, 6))



ax1 = sns.countplot(x="age_approx",

                    data = train,

                    order = train_ages.index, ax = ax[0])

ax1.set_title('Train', fontsize = 20, weight = 'bold')

    

ax2 = sns.countplot(x="age_approx",

                    data=test,

                    order = train_ages.index, ax = ax[1])

ax2.set_title('Test', fontsize = 20, weight = 'bold')
sex_benign = train[train.target == 0].sex.dropna()

sex_malig = train[train.target == 1].sex.dropna()



sex = pd.DataFrame({'malignant':sex_malig.value_counts(),

                     'benign':sex_benign.value_counts(),

                     })
sex.plot(kind='bar',secondary_y = 'malignant',  figsize = (6, 4))

plt.title("Sex and Melanoma : Train", fontsize = 20, weight= 'bold')
fig, ax = plt.subplots(1,2, figsize = (6, 4))



ax1 = sns.countplot(x="sex",

                    data = train,

                    ax = ax[0])

ax1.set_title('Train', fontsize = 20, weight = 'bold')

    

ax2 = sns.countplot(x="sex",

                    data=test,

                    ax = ax[1])

ax2.set_title('Test', fontsize = 20, weight = 'bold')