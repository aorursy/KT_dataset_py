import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os
train = pd.read_csv('../input/janatahack/train_8wry4cB.csv')

train.head()
test = pd.read_csv('../input/janatahack/test_Yix80N0.csv')

test.head()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.set_style('whitegrid')

sns.countplot(x='gender',data=train)
train['gender'].value_counts()
train['ProductList'].value_counts()