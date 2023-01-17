# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
train.info()
train.head()
train.describe()
cols=train.columns

non_numeric_cols = [i for i in cols if train[i].dtype=='object']
print(non_numeric_cols)
numeric_cols=[i for i in cols if i not in non_numeric_cols]
print(numeric_cols)
#importing libraris for visualizations

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train_clean=train[numeric_cols].dropna()



for num_col in train_clean.columns:

    plt.subplots(figsize=(12,10))

    sns.regplot(x=num_col,y='SalePrice',data=train_clean)

train.isnull().sum()
len(train)
train[['PoolQC','SalePrice']].dropna()



train[non_numeric_cols].head()
corr=train.corr()
len(numeric_cols)
cat_cols=[col for col in non_numeric_cols if train[col].nunique()<=6]
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
length=len(train)

missing_perc_cols=[col for col in train.columns if train[col].isnull().sum(axis=0)/length>0.4]

missing_perc_cols
import seaborn as sns

for col in missing_perc_cols:

    plt.subplots(figsize=(12,10))

    sns.countplot(x=col,data=train)