import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import os

import json

from pandas.io.json import json_normalize

from pprint import pprint

from PIL import Image

from IPython.display import display, HTML
breeds = pd.read_csv('../input/breed_labels.csv')

colors = pd.read_csv('../input/color_labels.csv')

states = pd.read_csv('../input/state_labels.csv')

train = pd.read_csv('../input/train/train.csv')

test = pd.read_csv('../input/test/test.csv')



train.head()
sns.distplot(train['AdoptionSpeed'], bins = 5, kde=False)
sns.distplot(train['Age'], bins = 100, kde=False)
train.Age.unique()
train[train['Age'] == 81]['PetID']
from PIL import Image

from io import BytesIO

import requests

image = Image.open("../input/train_images/2ba708d92-1.jpg")

image
train['Age'] = train['Age'].fillna( method='pad')
train.loc[train['Age'] > 20, 'Age'] = train['Age'].mean()
train['Age']
sns.distplot(train['Age'], bins = 20, kde=False)
breed_dict = dict(zip(breeds['BreedID'], breeds['BreedName']))
animal_dict = {1: 'Dog', 2: 'Cat'}
train.head()
def make_features(train):

    train['Named'] = (~train['Name'].isnull() | train['Name'].str.contains('(No | Puppies | Kitty | Puppy | Kitten)')).astype(int)

    train['PureBreed'] = (train.Breed2==0 | ~train.Breed1.isin(breeds[breeds.BreedName.str.contains('Domestic' or 'Mixed')].BreedID.values)).astype(int)

    train = train.drop('Breed2', axis=1)



    train = train.select_dtypes(exclude=[object])

    return train



dealed_train = make_features(train)
dealed_train['Type'] = dealed_train['Type'].map(animal_dict)

dealed_train['Breed1_new'] = dealed_train['Breed1'].map(breed_dict)
train_dogs = dealed_train[dealed_train['Type'] == 'Dog']

train_cats = dealed_train[dealed_train['Type'] == 'Cat']
train_dogs.Breed1_new.unique()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))

sns.countplot(data=train_dogs, x='PureBreed',hue='AdoptionSpeed', ax=ax1)

plt.title('Dogs adoption speed')

sns.countplot(data=train_cats, x='PureBreed',hue='AdoptionSpeed', ax=ax2)

plt.title('Cats adoption speed')
dealed_train.head()
train['Breed1_new'] = train['Breed1'].map(breed_dict)

train['Type'] = train['Type'].map(animal_dict)
train.groupby(['Type', 'Breed1_new', 'FurLength'])['PetID'].count()
train_dogs['Breed1_new'].value_counts().iloc[:5]
train_cats['Breed1_new'].value_counts().iloc[:5]
plt.figure(figsize=(14, 6));

sns.factorplot(x='Named', y='AdoptionSpeed', data=dealed_train, kind='bar')

plt.title('Having name is good or bad?')
plt.figure(figsize=(28, 12));

sns.barplot(x='PhotoAmt', y='AdoptionSpeed', data=train)

plt.title('What about photos?')
sns.lmplot(x='Fee',y='AdoptionSpeed',data=train,hue='Type')