# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd 



import os

print(os.listdir("../input"))



#
suicides='../input/suicide-rates-overview-1985-to-2016/master.csv'

suicides

data=pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')

data.head()

# Determining the size and shape of the data

data.shape

# Data types in the data

data.dtypes
# Missing values in the dataset

data.info()

# Which country has the highest suicide cases

# target variable country

y = data.year

y.head()

# predictors variables

features=['age','population','sex','suicides_no','country']

features

X = data[features]

M=pd.get_dummies(X)

M.head()
# summary statistics of the data

M.describe()
# Graphical visualization with seaborn

import seaborn as sn

sn.pairplot(X[['suicides_no','sex','population','age','country']])

# multivariate correlation

corrmat=X[['suicides_no','sex','population','age']].corr()

corrmat

mask=np.array(corrmat)

mask

mask[np.tril_indices_from(mask)]=False

sn.heatmap(corrmat,mask=mask,vmax=.8,square=True,annot=True)