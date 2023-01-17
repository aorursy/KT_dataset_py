#import necessary packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



#Read Data

train = pd.read_csv("/kaggle/input/home-data-for-ml-course/train.csv")

test = pd.read_csv("/kaggle/input/home-data-for-ml-course/test.csv")

#Having quick look at the data

train.head()
null_list = train.isnull().sum()

print(null_list[null_list != 0] )
null_list_test=test.isnull().sum()

print(null_list_test[null_list_test!=0])
check_fireplace = train[['Fireplaces','FireplaceQu']]

check_fireplace[check_fireplace['FireplaceQu'].isnull()].head()
valid_na_list = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']

train[null_list[valid_na_list].index].dtypes
test[null_list_test[valid_na_list].index].dtypes
#Checking the unique number of values in each

train[null_list[valid_na_list].index].nunique()
#Let's replace NA by 0

train[valid_na_list]=train[valid_na_list].fillna('0')

test[valid_na_list]=test[valid_na_list].fillna('0')

#validating the null list again

null_list=train.isnull().sum()

null_list_test=test.isnull().sum()

print('-----NA in train set-----')

print(null_list[null_list != 0])

print()

print('-----NA in test set-----')

print(null_list_test[null_list_test!=0])
#Now check the dtype of left features

train[null_list[null_list !=0].index].dtypes
mean_LF = round(train['LotFrontage'].mean(),2)

train['LotFrontage'] = train['LotFrontage'].fillna(mean_LF)



mean_GB = round(train['GarageYrBlt'].mean(),2)

train['GarageYrBlt'] = train['GarageYrBlt'].fillna(mean_GB)



#validating the null list again

null_list=train.isnull().sum()

null_list[null_list != 0]
train.dropna(0,inplace=True)

train.shape
#validating the null list again

null_list=train.dropna(0).isnull().sum()

null_list[null_list != 0]
train.duplicated().sum()
#List out all categorical features

cat_feats = train.columns[train.dtypes== object]

cat_feats
train[cat_feats].nunique()
feat_for_label = ['LotShape','LandContour','Utilities','LandSlope','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure',

'BsmtFinType1','BsmtFinType2','HeatingQC','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual',

'GarageCond','PoolQC','Fence']



feat_for_oneHot = ['MSZoning','Street','Alley','LotConfig','Condition1','Condition2','BldgType','HouseStyle','RoofStyle',

'RoofMatl','MasVnrType','Foundation','Heating','CentralAir','Electrical','PavedDrive','MiscFeature','SaleType','SaleCondition']
#Will perform Label Encoding

from sklearn.preprocessing import LabelEncoder

encoded_train=train[train.columns[train.dtypes != object]]



LE = LabelEncoder()

for feature in feat_for_label:

    encoded_train[feature] = LE.fit_transform(train[feature])



print(encoded_train.shape)

print(encoded_train.head())
#Will perform One-Hot Encoding

from sklearn.preprocessing import OneHotEncoder



OH = OneHotEncoder(sparse=False,handle_unknown='ignore')

OH_vals=pd.DataFrame(OH.fit_transform(train[feat_for_oneHot]),columns=OH.get_feature_names(feat_for_oneHot))



#OH was changing index(adding rows) so needed to reset_index, but reset_index adds new column so added drop=True to avoid that

encoded_train.reset_index(drop=True,inplace=True)

encoded_train=pd.concat([encoded_train,OH_vals],axis=1)

encoded_train.index = train.index



print(encoded_train.shape)

print(encoded_train.head())
obj_list = ['Neighborhood','Exterior1st','Exterior2nd']

from sklearn.feature_extraction import FeatureHasher

fh = FeatureHasher(n_features=6, input_type='string')

encoded_train.reset_index(drop=True,inplace=True)

for i in obj_list:

    hashed_features = fh.fit_transform(train[i])

    column = [i+'_1',i+'_2',i+'_3',i+'_4',i+'_5',i+'_6']

    hashed_features = pd.DataFrame(hashed_features.toarray(),columns=column)

    print(hashed_features.shape)

    encoded_train=pd.concat([encoded_train, hashed_features], 

              axis=1)

encoded_train.index=train.index

encoded_train.shape
encoded_train.head()
#is there any categorical column?

(encoded_train.dtypes == object).sum()
#Drop the ID column 

encoded_train.drop(axis=1,columns='Id',inplace=True)

encoded_train.describe()
for i in range(encoded_train.shape[1]):

    mean  = encoded_train.iloc[:,i].min()

    max_min = encoded_train.iloc[:,i].max() - encoded_train.iloc[:,i].min()

    temp = encoded_train.iloc[:,i]

    encoded_train.iloc[:,i] = round((temp - mean) / max_min,3)



encoded_train.describe()