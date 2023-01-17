%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



train = pd.read_csv('../input/train.csv',names=['id', 'spacegroup', 'natoms', 'al',

       'ga', 'in', 'a',

       'b', 'c',

       'alpha', 'beta',

       'gamma', 'E0',

       'bandgap'],header=0,sep=',')

test = pd.read_csv('../input/test.csv',names=['id', 'spacegroup', 'natoms', 'al',

       'ga', 'in', 'a',

       'b', 'c',

       'alpha', 'beta',

       'gamma'],header=0,sep=',')



full = pd.concat([train,test])
train['spacegroup'].value_counts(normalize=True).plot.bar()

plt.title('Spacegroup distribution in the training set')

plt.ylabel('% of crystals in spacegroup')

plt.xlabel('Space group')

plt.show()
test['spacegroup'].value_counts(normalize=True).plot.bar()

plt.title('Spacegroup distribution in the test set')

plt.ylabel('% of crystals in spacegroup')

plt.xlabel('Space group')

plt.show()
train[['a','b','c','alpha','beta','gamma']][train['spacegroup'] == 12].describe()
train[train['spacegroup'] == 12].hist(figsize=(12,8),column = ['a','b','c','alpha','beta','gamma'],layout =(2,3))

plt.show()
train[['a','b','c','alpha','beta','gamma']][train['spacegroup'] == 33].describe()
train[train['spacegroup'] == 33].hist(figsize=(12,8),column = ['a','b','c','alpha','beta','gamma'],layout =(2,3))

plt.show()
train[['a','b','c','alpha','beta','gamma']][train['spacegroup'] == 167].describe()
train[train['spacegroup'] == 167].hist(figsize=(12,8),column = ['a','b','c','alpha','beta','gamma'],layout =(2,3))

plt.show()
train[['a','b','c','alpha','beta','gamma']][train['spacegroup'] == 194].describe()
train[train['spacegroup'] == 194].hist(figsize=(12,8),column = ['a','b','c','alpha','beta','gamma'],layout =(2,3))

plt.show()
train[['a','b','c','alpha','beta','gamma']][train['spacegroup'] == 206].describe()
train[train['spacegroup'] == 206].hist(figsize=(12,8),column = ['a','b','c','alpha','beta','gamma'],layout =(2,3))

plt.show()
train[['a','b','c','alpha','beta','gamma']][train['spacegroup'] == 227].describe()
train[train['spacegroup'] == 227].hist(figsize=(12,8),column = ['a','b','c','alpha','beta','gamma'],layout =(2,3))

plt.show()
train.groupby(['spacegroup'])['al'].describe()
train.groupby(['spacegroup'])['ga'].describe()

train.groupby(['spacegroup'])['in'].describe()

train.groupby(['spacegroup'])['E0'].describe()



train[train['E0'] < 0.05].groupby(['spacegroup'])['E0'].describe()

train.groupby(['spacegroup'])['bandgap'].describe()
train[train['bandgap'] >= 3.2].groupby(['spacegroup'])['bandgap'].describe()

train[(train['bandgap'] >= 3.2) & (train['E0'] < 0.05)].groupby(['spacegroup'])['bandgap'].describe()

train[(train['bandgap'] >= 3.2) & (train['E0'] < 0.05)].groupby(['spacegroup'])['bandgap'].agg([ 'count'])/train[(train['E0'] < 0.05)].groupby(['spacegroup'])['bandgap'].agg([ 'count'])