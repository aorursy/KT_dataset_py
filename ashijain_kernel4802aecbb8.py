# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

import os

import gc

import psutil

import pandas as pd

import numpy as ny
sample_file=pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

sample_file.head()
train_file=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

train_file.head()
test_file=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

test_file.head()
train_file.columns
test_file.columns
sns.distplot(train_file['SalePrice'])
sns.kdeplot(train_file['SalePrice'])
train_file.info()
train_file.isnull().mean()
train_file.shape
train_file.describe()
train_new_file=train_file.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1)
train_new_file['MSSubClass'].unique()
train_new_file
sns.countplot(train_new_file['MSSubClass'])
sns.boxplot(x='MSSubClass', y='SalePrice',data=train_new_file)
sns.countplot(train_new_file['MSZoning'])
train_new_file['LotArea'].unique()
corr=train_new_file.corr()['SalePrice']

corr
corr=train_new_file.corr()['SalePrice']

relative_cols=[]

for k,v in corr.items():

    if((v>0.5) & (v<1)):

        relative_cols.append(k)        

relative_cols