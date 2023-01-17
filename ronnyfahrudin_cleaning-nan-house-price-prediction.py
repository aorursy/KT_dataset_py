# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
trainset = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

testset = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
#build function to count caolumn that has missing value

def column_nan(missing):

    s = []

    for i in missing:

        if i > 0:

            s.append(i)

    print(len(s))

train = trainset.isnull().sum().sort_values(ascending = False)

test = testset.isnull().sum().sort_values(ascending = False)
print("Sum of column has Nan in trainset:")

column_nan(train)

print("Sum of column has Nan in testset:")

column_nan(test)
#summing missing value trainset

trainset.isnull().sum().sort_values(ascending = False).head(20)/len(trainset)
#summing percentage missing value

testset.isnull().sum().sort_values(ascending = False).head(33)/len(trainset)
#clean columns 'PoolQC','MiscFeature','Alley','Fence' because columns has missing values more than 80%

cleaning_train = trainset.drop(['PoolQC','MiscFeature','Alley','Fence'], axis= 1)

cleaning_test = testset.drop(['PoolQC','MiscFeature','Alley','Fence'], axis= 1)
#cleaning trainig_set

s = cleaning_train.isnull().sum(axis=0).reset_index().sort_values(0,ascending=False).head(15)

s.columns = ['variable','missing']

col_miss_train = s['variable'].tolist()

miss = cleaning_train[col_miss_train]

miss.describe()
# see distribution

fig,axes = plt.subplots(1,3, figsize = (20,6))

sns.distplot(miss['LotFrontage'], color = 'b',ax = axes[0])

sns.distplot(miss['GarageYrBlt'], color = 'r', ax = axes[1])

sns.distplot(miss['MasVnrArea'], color = 'y',ax = axes[2])
# clean with median on numerical

cleaning_train[['LotFrontage','GarageYrBlt','MasVnrArea']] = cleaning_train[['LotFrontage','GarageYrBlt','MasVnrArea']].fillna(cleaning_train[['LotFrontage','GarageYrBlt','MasVnrArea']].median())
# clean with mode on categorical

list_miss = cleaning_train.isnull().sum().sort_values(ascending = False).head(12).index.values.tolist()

cleaning_train[list_miss] = cleaning_train[list_miss].fillna(cleaning_train[list_miss].mode().iloc[0])
# Cleaning testset in numerical

list_miss_test = cleaning_test.isnull().sum().sort_values(ascending = False).head(29).index.values.tolist()

list_numeric = cleaning_test[list_miss_test].describe().columns.values.tolist()

# handling with median

cleaning_test[list_numeric] = cleaning_test[list_numeric].fillna(cleaning_test[list_numeric].median())
# cleaning testset in categorical

lst_categ = cleaning_test.isnull().sum().sort_values(ascending = False).head(18).index.values.tolist()

cleaning_test[lst_categ] = cleaning_test[lst_categ].fillna(cleaning_test[lst_categ].mode().iloc[0])
cleaning_train.isnull().sum().sort_values()
cleaning_test.isnull().sum().sort_values()