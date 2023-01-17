import numpy as np

import pandas as pd

from IPython.display import clear_output

import matplotlib.pyplot as plt
train = pd.read_csv('../input/train.csv')
#train.describe()

clear_output()
train.info()

clear_output()

#填充缺失值

train['LotFrontage'].fillna(train['LotFrontage'].mean(), inplace = True)
train['MSZoning'].unique()

#对训练集的MSZoning进行one-hot编码

train = train.join(pd.get_dummies(train['MSZoning']))

train.info()
train.drop(['MSZoning'], axis=1, inplace=True)

#train.info()

#clear_output()
train =train.join(pd.get_dummies(train['Street']))

train.drop(['Street'], axis = 1, inplace = True)
train['Alley'].unique()

train['Alley'].value_counts()

train.drop(['Alley'], axis =1, inplace = True)
train['LotShape'].unique()

train['LotShape'].value_counts()

train =train.join(pd.get_dummies(train['LotShape']))

train.drop(['LotShape'], axis = 1, inplace = True)
train['LandContour'].value_counts()

train =train.join(pd.get_dummies(train['LandContour']))

train.drop(['LandContour'], axis = 1, inplace = True)

#clear_output()
train.drop(['Utilities'], axis = 1, inplace = True)
train['LotConfig'].value_counts()