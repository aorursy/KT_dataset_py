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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

import missingno as msno

import warnings

warnings.filterwarnings("ignore")

df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
df_train.head()
df_test.head()
df_train_missing = df_train.isna().sum()[df_train.isna().sum()!=0].sort_values(ascending=True)

#df_train_missing.columns = ['Missing_Count']

#df_train_missing['Missing_Count_Percent'] = df_train.isna().sum()[df_train.isna().sum()!=0]/len(df_train)
df_test_missing = df_test.isna().sum()[df_test.isna().sum()!=0].sort_values(ascending=True)

#df_test_missing.columns = ['Missing_Count']

#df_test_missing['Missing_Count_Percent'] = df_test.isna().sum()[df_test.isna().sum()!=0]/len(df_test)
#df_train = df_train.drop(df_train_missing[df_train_missing['Missing_Count_Percent']>0.1].index, axis=1)
#df_test = df_test.drop(df_test_missing[df_test_missing['Missing_Count_Percent']>0.1].index, axis=1)
df_train.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','LotFrontage'],axis=1,inplace=True)

df_test.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','LotFrontage'],axis=1,inplace=True)
df_train_missing.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','LotFrontage'],axis=0,inplace=True)

df_test_missing.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','LotFrontage'],axis=0,inplace=True)
for cols in list(df_train_missing.index):

    if df_train[cols].dtypes == 'object':

        df_train[cols].fillna(df_train[cols].value_counts().index[0], inplace=True)

    elif df_train[cols].dtypes in ['float64', 'int64']:

        df_train[cols].fillna(df_train[cols].median(), inplace=True)
for cols in list(df_test_missing.index):

    if df_test[cols].dtypes == 'object':

        df_test[cols].fillna(df_test[cols].value_counts().index[0], inplace=True)

    elif df_test[cols].dtypes in ['float64', 'int64']:

        df_test[cols].fillna(df_test[cols].median(), inplace=True)
set(df_train.columns)-set(df_test.columns)
mismatch_columns = []



for col in df_train.columns:

    if df_train[col].dtype == "object":

        if df_train[col].nunique() != df_test[col].nunique():

            mismatch_columns.append(col)
mismatch_columns
data = pd.concat([df_train['SalePrice'], df_train['OverallQual']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x='OverallQual', y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
data1 = pd.concat([df_train['SalePrice'], df_train['GrLivArea']], axis=1)

data1.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,800000));
data2 = pd.concat([df_train['SalePrice'], df_train['OverallCond']], axis=1)

f, ax = plt.subplots(figsize=(10,5))

fig = sns.boxplot(x='OverallCond', y='SalePrice', data=data2)

fig.axis(ymin=0, ymax=800000);
data3 = pd.concat([df_train['SalePrice'], df_train['YearBuilt']], axis=1)

f, ax = plt.subplots(figsize=(12,7))

fig = sns.boxplot(x='YearBuilt', y='SalePrice', data=data3)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=80)
data4 = pd.concat([df_train['SalePrice'], df_train['HouseStyle']], axis=1)

f, ax = plt.subplots(figsize=(10,5))

fig = sns.boxplot(x='HouseStyle', y='SalePrice', data=data4)

fig.axis(ymin=0, ymax=800000);
data5 = pd.concat([df_train['SalePrice'], df_train['TotalBsmtSF']], axis=1)

data5.plot.scatter(x='TotalBsmtSF', y='SalePrice', ylim=(0,800000));
data6 = pd.concat([df_train['SalePrice'], df_train['TotRmsAbvGrd']], axis=1)

f, ax = plt.subplots(figsize=(10,5))

fig = sns.boxplot(x='TotRmsAbvGrd', y='SalePrice', data=data6)

fig.axis(ymin=0, ymax=800000);
sns.distplot(df_train['SalePrice'])
df_train.drop(df_train.index[[691,1182,1169,898]], inplace=True)
df_train['SalePrice'] = np.log(df_train['SalePrice'])

sns.distplot(df_train['SalePrice'])
object_cols = list(df.select_dtypes(exclude=[np.number]).columns)

object_cols_ind = []

for col in object_cols:

    object_cols_ind.append(df.columns.get_loc(col))



# Encode the categorical columns with numbers    

label_enc = LabelEncoder()

for i in object_cols_ind:

    df.iloc[:,i] = label_enc.fit_transform(df.iloc[:,i])