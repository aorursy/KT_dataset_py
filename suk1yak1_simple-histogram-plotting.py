import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
base_path = '/kaggle/input/liverpool-ion-switching/'
%%time

train = pd.read_csv(base_path + 'train.csv')

test = pd.read_csv(base_path + 'test.csv')
plt.figure(figsize=(10,5))

plt.title(f'train all')

sns.distplot(train['signal'])

plt.show()

plt.figure(figsize=(10,5))

plt.title(f'train devide by target')

for i in range(11):

    sns.distplot(train.loc[train['open_channels']==i,'signal'],label=i)

plt.legend()

plt.show()

plt.figure(figsize=(10,5))

plt.title(f'test all')

sns.distplot(test['signal'])

plt.show()
train['batch_id'] = np.ceil(train['time'] / 50).astype(int)

test['batch_id'] = np.ceil(test['time'] / 50).astype(int)
for batch_id in set(train['batch_id']):

    plt.figure(figsize=(10,5))

    plt.title(f'batch_id:{batch_id} all')

    sns.distplot(train.loc[(train['batch_id']==batch_id) ,'signal'])

    plt.show()

    

    plt.figure(figsize=(10,5))

    plt.title(f'batch_id:{batch_id} devide by target')

    for i in range(11):

        sns.distplot(train.loc[(train['open_channels']==i) & (train['batch_id']==batch_id) ,'signal'],label=i)

    plt.legend()

    plt.show()
for batch_id in set(test['batch_id']):

    plt.figure(figsize=(10,5))

    plt.title(f'batch_id:{batch_id} all')

    sns.distplot(test.loc[(test['batch_id']==batch_id) ,'signal'])

    plt.show()