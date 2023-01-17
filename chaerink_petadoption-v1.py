import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

import missingno as msno

os.listdir('../input/petfinder-adoption-prediction')
train = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv')

train.head()
train.columns
train.groupby('Type')['AdoptionSpeed'].mean()
train['Age_yr'] = train['Age'].apply(lambda x: x // 12)

train['Age_yr'][:5]
train[train['Age_yr']>0].head().style.background_gradient(cmap='Pastel1_r')
train.groupby('Age_yr')['AdoptionSpeed'].mean()
train.groupby('Gender')['AdoptionSpeed'].mean()
train.groupby('FurLength')['AdoptionSpeed'].mean()
train.groupby('FurLength')['AdoptionSpeed'].count()
train.groupby('AdoptionSpeed')['Age'].count()
train.groupby(['Vaccinated', 'Dewormed','Sterilized'])['AdoptionSpeed'].mean()
train.groupby('PhotoAmt')['AdoptionSpeed'].mean()
train['Photo_yn'] = train['PhotoAmt'].apply(lambda x : np.where(x>0, 1, 0))

train.groupby('Photo_yn')['AdoptionSpeed'].mean()
train['Description'].iloc[:5]
def lent(x):

    if type(x) != str:

        return 1

    else:

        t = x.split()

        return len(t)

    

train['Description_length'] = train['Description'].apply(lent)

train['Description_length'].sample(5)
train['Description_length'].describe()
plt.figure(figsize=(12,8))

plt.hist(train['Description_length'], bins=300, density=True, color='plum', orientation='vertical')

plt.title('Description Length')

plt.show()
train['Description_length_class'] = train['Description_length'].apply(lambda x: x//20)

x = train.groupby('Description_length_class')['AdoptionSpeed'].mean()

ind = x.index



plt.figure(figsize=(12,8))

plt.bar(ind, x, color='plum')

plt.title('Length - Adoption Speed')

plt.show()