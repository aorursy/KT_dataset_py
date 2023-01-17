# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data=pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')

test_data=pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')
test_data.columns
'SalePrice' in train_data
test_data.shape
num_col=train_data.select_dtypes(exclude='object')

cat_col=train_data.select_dtypes(exclude=['int64','float64'])
# HEATMAP TO SEE MISSING VALUES

plt.figure(figsize=(15,5))

sns.heatmap(num_col.isnull(),yticklabels=0,cbar=False,cmap='viridis')
num_col.columns
y=train_data['SalePrice'] # or "y = num_col.SalePrice" # storing target variable in y

y
num_col.isna().sum().reset_index() 
f,ax = plt.subplots(figsize=(25,2))

sns.heatmap(num_col.corr().iloc[1:2,:], annot=True, linewidths=.8, fmt= '.1f',ax=ax)
sns.kdeplot(num_col.LotFrontage,Label='LotFrontage',color='g')
num_col.LotFrontage.describe()
train_data['LotFrontage'].replace({np.nan:train_data.LotFrontage.mean()},inplace=True)

print("Done!")
sns.kdeplot(num_col['MasVnrArea'],Label='MasVnrArea',color='g')
num_col.MasVnrArea.value_counts()
num_col.MasVnrArea.describe()
train_data.MasVnrArea.replace({np.nan:0},inplace=True)

print("Done!")
train_data.MasVnrArea.isnull().any().sum()
num_col.MasVnrArea.isnull().any().sum()
f,ax = plt.subplots(figsize=(20,2))

sns.heatmap(num_col.corr().iloc[24:25,:], annot=True, linewidths=.8, fmt= '.1f',ax=ax)
sns.kdeplot(num_col.GarageYrBlt,Label='GarageYrBlt',color='g')
train_data['GarageYrBlt'].describe()
train_data['GarageYrBlt'] = train_data['GarageYrBlt'].fillna(num_col.GarageYrBlt.mean())

print("Done!")
train_data.GarageYrBlt.isnull().any().sum()
len(cat_col.columns)
Nan_cat_col = [col for col in cat_col.select_dtypes(exclude=['int64','float64']).columns 

           if cat_col.select_dtypes(exclude=['int64','float64'])[col].isna().any()]

len(Nan_cat_col)
Nan_cat_col
len(train_data.Id)
cat_col[Nan_cat_col].isnull().sum().reset_index()
'Alley' in cat_col[Nan_cat_col].columns
for col in ['Alley','PoolQC','Fence','MiscFeature'] :

    Nan_cat_col.remove(col)
train_data = train_data.drop(['Alley','PoolQC','Fence','MiscFeature'] , axis=1)

print("Done!")
'MiscFeature' in cat_col
len(cat_col.columns)
fig = plt.figure(figsize=(18,16))

for index,col in enumerate(train_data[Nan_cat_col]):

    sns.catplot(x=col, y="SalePrice", kind="box" , data=train_data)

fig.tight_layout(pad=1.0)
fig = plt.figure(figsize=(18,20))

for index in range(len(cat_col.columns)):

    plt.subplot(9,5,index+1)

    sns.countplot(x=cat_col.iloc[:,index] , data=cat_col.dropna())

    plt.xticks(rotation=90)

fig.tight_layout(pad=1.0)
cat_col.columns
train_data = train_data.drop(['Street','LandContour','Utilities','Condition1','Condition2','LandSlope',

                     'RoofMatl','Functional','GarageCond','Heating','GarageQual'],axis=1)

print("Done!")
'Alley' in train_data.columns
for col in ['Street','LandContour','Utilities','Condition1','Condition2','LandSlope',

                     'RoofMatl','Functional','GarageCond','Heating','GarageQual'] :

    if col in Nan_cat_col :

        Nan_cat_col.remove(col)

len(train_data[Nan_cat_col].columns)
train_data[Nan_cat_col].columns
for col in train_data[Nan_cat_col].columns :

   train_data[col].replace({np.nan:train_data[col].value_counts().idxmax()},inplace=True)

print("Done")
train_data.loc[:,'MSSubClass':'SaleCondition']
train_data.head()
test_data.head()
num_col_T =test_data.select_dtypes(exclude='object')

cat_col_T =test_data.select_dtypes(exclude=['int64','float64'])
num_col_T.isna().sum().reset_index() 
test_data = test_data.drop(['Alley','PoolQC','Fence','MiscFeature'] , axis=1)

print("Done!")
test_data = test_data.drop(['Street','LandContour','Utilities','Condition1','Condition2','LandSlope',

                     'RoofMatl','Functional','GarageCond','Heating','GarageQual'],axis=1)

print("Done!")
sns.kdeplot(num_col_T.LotFrontage,Label='LotFrontage',color='g')
test_data['LotFrontage'].describe()
test_data['LotFrontage'] = test_data['LotFrontage'].fillna(test_data['LotFrontage'].describe()["mean"])

test_data['LotFrontage'].isnull().any()
sns.kdeplot(num_col_T.LotFrontage,Label='MasVnrArea',color='b')
test_data['MasVnrArea'].describe()
test_data.MasVnrArea = test_data.MasVnrArea.replace({np.nan:0},inplace=True)

print("Done!")
test_data['GarageYrBlt'].describe()
test_data['GarageYrBlt'] = train_data['GarageYrBlt'].fillna(num_col.GarageYrBlt.mean())

print("Done!")
test_data['BsmtFinSF1'].describe()['mean']
col_to_edit = ['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath',

            'BsmtHalfBath','GarageCars','GarageArea']

for col in col_to_edit :

    test_data[col] = test_data[col].replace({np.nan:test_data[col].mean()},inplace=True)
num_col_T.columns.isna().any()
test_data['MasVnrArea'].dtypes
test_data['MasVnrArea']
test_data['MasVnrArea'] = test_data['MasVnrArea'].apply(pd.to_numeric, errors='coerce')

test_data['MasVnrArea']
cat_col_T =test_data.select_dtypes(exclude=['int64','float64'])

cat_col_T.isna().sum().reset_index() 
test_data['MSZoning'].mode()[0]
for col in ['MSZoning','Exterior1st','Exterior2nd',"MasVnrType",'SaleType','KitchenQual'] :

    test_data[col] = test_data[col].fillna(test_data[col].mode()[0])
cat_col_T =test_data.select_dtypes(exclude=['int64','float64'])

for col in cat_col_T.columns :

    if cat_col_T[col].isna().sum() > 0 :

        print(col)
train_data.SalePrice
train_data.SalePrice.describe()
train_data.SalePrice.describe()['50%']
train_data.SalePrice.loc[train_data.Id == 1455].reset_index()['SalePrice']
train_data.SalePrice.describe()['min'] > train_data.SalePrice.loc[train_data.Id == 1455]
(train_data.SalePrice.describe()['min'] > train_data.SalePrice.loc[train_data.Id == 1455]) == False
train_data.columns.get_loc('SalePrice')
max(list(train_data.Id))
train_data.stack()[44]['SalePrice']
if (train_data.SalePrice.describe()['50%'] >= train_data.stack()[65]['SalePrice']) == False :

    print(1)
train_data.SalePrice.describe()['min'] <= train_data.stack()[1454]['SalePrice']
1460 in list(train_data.Id)[0:-1]
Id_For_N_P = []

Id_For_H_P = []

Id_For_VH_P = []

for i in list(train_data.Id)[0:-1] :

    if train_data.SalePrice.describe()['min'] < train_data.stack()[i]['SalePrice'] and train_data.SalePrice.describe()['50%'] >= train_data.stack()[i]['SalePrice'] :

        Id_For_N_P.append(i)

    elif train_data.SalePrice.describe()['50%'] < train_data.stack()[i]['SalePrice'] and train_data.SalePrice.describe()['75%'] >= train_data.stack()[i]['SalePrice'] :

        Id_For_H_P.append(i)

    elif train_data.SalePrice.describe()['75%'] < train_data.stack()[i]['SalePrice'] and train_data.SalePrice.describe()['max'] >= train_data.stack()[i]['SalePrice'] :

        Id_For_VH_P.append(i)
Id_For_N_P
train_data.stack()[1400][['BsmtQual','BsmtCond']]