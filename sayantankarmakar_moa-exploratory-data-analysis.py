import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



import torch

import torch.nn as nn



%matplotlib inline
sample_submission = pd.read_csv("/kaggle/input/lish-moa/sample_submission.csv")

train_targets_scored = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")

train_targets_nonscored = pd.read_csv("/kaggle/input/lish-moa/train_targets_nonscored.csv")

train_features = pd.read_csv("/kaggle/input/lish-moa/train_features.csv")

test_features = pd.read_csv("/kaggle/input/lish-moa/test_features.csv")
print("train_features: ", train_features.shape)

display(train_features.head())
print("test_features: ", test_features.shape)

display(test_features.head())
print("train_features description")

train_features.describe()
print("test_features description")

test_features.describe()
features = pd.concat([train_features, test_features])

features['train'] = [1]*len(train_features)+[0]*len(test_features)



g_cols = [c for c in features.columns if c.startswith('g-')]

c_cols = [c for c in features.columns if c.startswith('c-')]

cat_cols = ['cp_type', 'cp_time', 'cp_dose']



print("g_cols: ", len(g_cols))

print("c_cols: ", len(c_cols))
fig, axes = plt.subplots(nrows=8, ncols=2, figsize=(20,32))

for ax in axes.reshape(-1):

    c = g_cols[np.random.randint(0, len(g_cols))]

    sns.distplot(train_features[c], hist=True, ax=ax, bins=30, color='blue')

    sns.distplot(test_features[c], hist=True, ax=ax, bins=30, color='red')

plt.show()
fig, axes = plt.subplots(nrows=8, ncols=2, figsize=(20,32))

for ax in axes.reshape(-1):

    c = c_cols[np.random.randint(0, len(c_cols))]

    sns.distplot(train_features[c], hist=True, ax=ax, bins=30, color='orange')

    sns.distplot(test_features[c], hist=True, ax=ax, bins=30, color='mediumspringgreen')

plt.show()
fig, axes = plt.subplots(1,3, figsize=(20,6))



for i,ax in enumerate(axes.reshape(-1)):

    sns.countplot(x=cat_cols[i], hue='train', data=features, ax=ax, palette='BuGn')

plt.show()