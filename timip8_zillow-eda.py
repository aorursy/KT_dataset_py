import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
properties = pd.read_csv('../input/properties_2016.csv')

train = pd.read_csv('../input/train_2016.csv')
properties.head()
train.head()
properties.describe()
train.head()
print(train.size)

print(len(train['parcelid'].unique()))

print(len(train['transactiondate'].unique()))
train['transactiondate'].unique()
train['parcelid'].value_counts()
train['transactiondate'] = train['transactiondate'].apply(lambda x: x[0:4]+x[5:7])