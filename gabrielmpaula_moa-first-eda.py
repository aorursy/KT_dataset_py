import os

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
DATA_PATH = '../input/lish-moa/'
train_features = pd.read_csv(DATA_PATH + 'train_features.csv')

train_target = pd.read_csv(DATA_PATH + 'train_targets_scored.csv')
print('Target shape:', train_target.shape)

print(train_target.info())

train_target.head()
train_target.plot(kind='hist', legend=None, figsize=(20,8), title='Target values')

plt.show()
train_target.sum(axis=1).sort_values()
train_target.corr()
print('Shape:',train_features.shape)

print(train_features.info())

train_features.head()
plt.figure(figsize=(25,8))

columns = ['cp_dose', 'cp_type', 'cp_time']

for i, c in enumerate(columns, 1):

    plt.subplot(1,3,i)

    train_features[c].value_counts().plot(kind='bar', title=c)

plt.show()
g_columns = [c for c in train_features.columns if c.find('g-') > -1]



plt.figure(figsize=(25,15))

functions = {'Mean': np.mean,

             'Standard deviation': np.std,

             'Minimum': np.min,

             'Maximum': np.max}



for i, (name, function) in enumerate(functions.items(),1):

    plt.subplot(2,2,i)

    function(train_features[g_columns]).plot(kind='hist', title=name)
c_columns = [c for c in train_features.columns if c.find('c-') > -1]



plt.figure(figsize=(25,15))

functions = {'Mean': np.mean,

             'Standard deviation': np.std,

             'Minimum': np.min,

             'Maximum': np.max}



for i, (name, function) in enumerate(functions.items(),1):

    plt.subplot(2,2,i)

    function(train_features[c_columns]).plot(kind='hist', title=name)




plt.figure(figsize=(20,10))

train_features[g_columns].mean(axis=0).plot(kind='hist', alpha=0.5)

train_features[g_columns].median(axis=0).plot(kind='hist', alpha=0.5)

plt.title('Average and Median of g-Features')

plt.show()
c_columns = [c for c in train_features.columns if c.find('c-') > -1]



plt.figure(figsize=(20,10))

train_features[c_columns].mean(axis=0).plot(kind='hist', alpha=0.5)

train_features[c_columns].median(axis=0).plot(kind='hist', alpha=0.5)

plt.title('Average and Median of c-Features')

plt.show()