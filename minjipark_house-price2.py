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
import matplotlib.pyplot as plt

import seaborn as sns 

import numpy as np 
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
print("train 로우 수 = {} ".format(train.shape[0])) #행의 갯수

print("train 컬럼 수 = {} ".format(train.shape[1])) #열의 갯수



print("test 로우 수 = {} ".format(test.shape[0])) #행의 갯수

print("test 컬럼 수 = {} ".format(test.shape[1])) #열의 갯수
round(train.isnull().sum()[train.isnull().sum()>0]/len(train) * 100, 2).sort_values(ascending=False)
train_delete_columnidx = round(train.isnull().sum()[train.isnull().sum()>0]/len(train) * 100, 2).sort_values(ascending=False)[0:5].index

test_delete_columnidx = round(test.isnull().sum()[test.isnull().sum()>0]/len(train) * 100, 2).sort_values(ascending=False)[0:5].index
print(train_delete_columnidx)

print(test_delete_columnidx)
train = train.drop(train_delete_columnidx, axis=1)

train = train.drop(columns="Id")

test = test.drop(test_delete_columnidx, axis=1)

test_Id = test.Id

test = test.drop(columns='Id')
train_y = train.SalePrice

train = train.drop(columns='SalePrice')



total = pd.concat ([train, test], axis=0)
total.shape
categorical_index = total.dtypes[train.dtypes == "object"].index

numeric_index = total.dtypes[train.dtypes != "object"].index
categorical_null_idx = total[categorical_index].isnull().sum()[total[categorical_index].isnull().sum()>0].sort_values(ascending=False).index

categorical_null_idx
numeric_null_idx = total[numeric_index].isnull().sum()[total[numeric_index].isnull().sum()>0].sort_values(ascending=False).index

numeric_null_idx
corrmat = total.corr()

plt.subplots(figsize=(20,9))

sns.heatmap(corrmat, annot=True)

plt.show()
total[numeric_null_idx].corr()[total[numeric_null_idx].corr()>abs(0.6)]
just_mean_col = ['LotFrontage', 'GarageYrBlt', 'MasVnrArea', 'BsmtHalfBath', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF2']



for i in just_mean_col:

    total[i] = total[i].fillna(total[i].mean())
total.BsmtFullBath = total.BsmtFullBath.fillna(total.groupby(['BsmtFinSF1']).BsmtFullBath.mean())

total.GarageArea = total.BsmtFullBath.fillna(total.groupby(['GarageCars']).BsmtFullBath.mean())

total.GarageCars = total.BsmtFullBath.fillna(total.groupby(['GarageArea']).BsmtFullBath.mean())

total.BsmtFinSF1 = total.BsmtFullBath.fillna(total.groupby(['BsmtFullBath']).BsmtFullBath.mean())
total[numeric_null_idx].isnull().sum()
total[categorical_index].isnull().sum()[total[categorical_index].isnull().sum()>0].sort_values(ascending=False)
for i in categorical_null_idx:

    total[i].fillna(total[i].value_counts().idxmax(), inplace=True)
total.isnull().sum()
from sklearn import preprocessing



le = preprocessing.LabelEncoder()



for i in categorical_index:

    le.fit(total[i])

    total[i] = le.transform(total[i])
total.head()
total.shape
print(train.shape[0])
pretty_train = total[0:1460]

print(pretty_train.shape)
raw_train = pd.read_csv("../input/train.csv")

pretty_train2 = pd.concat([pretty_train, raw_train.SalePrice], axis=1)
pretty_train2.shape
pretty_train2.to_csv("after_cleaning.csv")