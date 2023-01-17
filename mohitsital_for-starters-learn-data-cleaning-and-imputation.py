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
# reading train and test data



train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')



print(train.shape)

print(test.shape)
##  train numerical columns and categorical columns

num_columns = []

cat_columns = []

for col in train.columns:

    if (train[col].dtypes != 'object'):

        num_columns.append(col)

    else:

        cat_columns.append(col)

        

print(num_columns)

print(cat_columns)
# Target variable

SalePrice = train['SalePrice']

# droppping target variable from train

train = train.drop('SalePrice',axis=1)

# adding marker variable to train and test dataset 

train['Marker'] = 1

test['Marker'] = 0
# appendng test data to train

fulldata = train.append(test).reset_index()

# shape of fulldata

print(fulldata.shape)

# checking for null values

fulldata.isnull().any()

sum(fulldata.isnull().any()) # 34 columns have null values



#d = np.where(fulldata.isnull().sum(axis=1)>1)

#df= test.drop(df.index[d])

#print(round(100*(1-test.count()/len(df)),2))



fulldata.isnull().sum() # number of null values per columns

(fulldata.isnull().sum() / len(fulldata))*100 # % of null values per columns

# dividing column in dictionary with different level of percentage of null values

lvl3_null = {}

lvl2_null = {}

lvl1_null = {}

for col in fulldata.columns:

    if ((fulldata[col].isnull().sum() / len(fulldata))*100) >= 80 :

        lvl3_null[col] = ((fulldata[col].isnull().sum() / len(fulldata))*100).round(3)

    elif ((fulldata[col].isnull().sum() / len(fulldata))*100) <= 10 :

        lvl1_null[col] = ((fulldata[col].isnull().sum() / len(fulldata))*100).round(3)

    else:

        lvl2_null[col] = ((fulldata[col].isnull().sum() / len(fulldata))*100).round(3)

        
print(lvl3_null)
print(lvl2_null)
print(lvl1_null)
################# doing imputation for lvl_3 null columns ##############################

#'Alley': 93.21685508735868,

#'PoolQC': 99.65741692360398,

#'Fence': 80.4385063377869,

#'MiscFeature': 96.40287769784173

        

#1

fulldata['Alley'].unique() # array([nan, 'Grvl', 'Pave'], dtype=object)

fulldata['Alley'].value_counts() # Name: Alley, dtype: int64

fulldata['Alley']= fulldata['Alley'].fillna('Noinfo') #  check for error or better error free imputing way

#2

fulldata['PoolQC'].value_counts()

fulldata = fulldata.drop('PoolQC', axis=1)

#3

fulldata['Fence'].value_counts()

fulldata['Fence']= fulldata['Fence'].fillna('Noinfo')

#4

fulldata['MiscFeature'].value_counts()

fulldata['MiscFeature']= fulldata['MiscFeature'].fillna('Nomisc')

################# doing imputation for lvl_2 null columns ##############################

#'LotFrontage': 16.649537512846866

#'FireplaceQu': 48.646796848235695



### imputing value of MSZoning as data will be grouped by MSzoning to fill null values of LotFrontage

fulldata['MSZoning'].isnull().sum()

fulldata['MSZoning'].value_counts()

fulldata['MSZoning']= fulldata['MSZoning'].fillna('RL')



#1

fulldata['LotFrontage'].value_counts()

id = fulldata['Id'][fulldata['LotFrontage'].isnull()]

fulldata['LotFrontage'] = fulldata['LotFrontage'].fillna(fulldata.groupby('MSZoning')['LotFrontage'].transform('mean').round())

#2  

fulldata['FireplaceQu'].value_counts()

fulldata['FireplaceQu']= fulldata['FireplaceQu'].fillna('Nofireplace')

################# doing imputation for lvl_1 null columns ##############################



 #'MSZoning': 0.137, 'Utilities': 0.069, 'Exterior1st': 0.034, 'Exterior2nd': 0.034,

 #'MasVnrType': 0.822, 'MasVnrArea': 0.788, 'BsmtQual': 2.775, 'BsmtCond': 2.809,

 #'BsmtExposure': 2.809, 'BsmtFinType1': 2.706, 'BsmtFinSF1': 0.034, 'BsmtFinType2': 2.741,

 #'BsmtFinSF2': 0.034, 'BsmtUnfSF': 0.034, 'TotalBsmtSF': 0.034, 'Electrical': 0.034,

 #'BsmtFullBath': 0.069, 'BsmtHalfBath': 0.069, 'KitchenQual': 0.034, 'Functional': 0.069,

 #'GarageType': 5.379, 'GarageYrBlt': 5.447, 'GarageFinish': 5.447, 'GarageCars': 0.034,

# 'GarageArea': 0.034, 'GarageQual': 5.447, 'GarageCond': 5.447, 'SaleType': 0.034,



# filling with dominant class

#fulldata['Utilities'].fillna(fulldata['Utilities'].value_counts().index[0],inplace=True)





mode_fill = ['Utilities','GarageYrBlt', 'Exterior1st', 'Exterior2nd','MasVnrType', 'MasVnrArea', 'BsmtFinSF1','BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Electrical','BsmtFullBath', 'BsmtHalfBath', 'KitchenQual', 'Functional','GarageCars','GarageArea',  'SaleType']



for col in mode_fill:

    fulldata[col].fillna(fulldata[col].value_counts().index[0],inplace=True)



#'BsmtQual': 2.775, 'BsmtCond': 2.809,'BsmtExposure': 2.809, 'BsmtFinType1': 2.706,'BsmtFinType2': 2.741



fulldata['BsmtQual']= fulldata['BsmtQual'].fillna('nobsmt')

fulldata['BsmtCond']= fulldata['BsmtCond'].fillna('nobsmt')

fulldata['BsmtExposure']= fulldata['BsmtExposure'].fillna('nobsmt')

fulldata['BsmtFinType1']= fulldata['BsmtFinType1'].fillna('nobsmt')

fulldata['BsmtFinType2']= fulldata['BsmtFinType2'].fillna('nobsmt')



#'GarageType': 5.379, 'GarageYrBlt': 5.447, 'GarageFinish': 5.447, 'GarageQual': 5.447, 'GarageCond': 5.447,



fulldata['GarageType']= fulldata['GarageType'].fillna('nogrg')

fulldata['GarageFinish']= fulldata['GarageFinish'].fillna('nogrg')

fulldata['GarageQual']= fulldata['GarageQual'].fillna('nogrg')

fulldata['GarageCond']= fulldata['GarageCond'].fillna('nogrg')

# seperating train and test data



dtrain = fulldata[fulldata['Marker']==1]

dtrain= dtrain.drop('index',axis=1)

dtest = fulldata[fulldata['Marker']==0]

dtest= dtest.drop('index',axis=1)

# removing outlier if any from num_columns

# LotFrontage

dtrain.boxplot(column=['LotFrontage']) # only two is above 200

dtest.boxplot(column=['LotFrontage']) # all are under 200

dtrain.drop(dtrain[dtrain['LotFrontage'] > 225].index, inplace = True) 

# LotArea

dtrain.boxplot(column=['LotArea']) # 

dtest.boxplot(column=['LotArea']) # all are under 100000

dtrain.drop(dtrain[dtrain['LotArea'] > 100000].index, inplace = True) 

# YearBuilt

dtrain.boxplot(column=['YearBuilt']) # few houses before yr 1880

dtest.boxplot(column=['YearBuilt']) # few houses before yr 1880

#YearRemodAdd

dtrain.boxplot(column=['YearRemodAdd'])

dtest.boxplot(column=['YearRemodAdd']) # no outliers

# YrSold

dtrain.boxplot(column=['YrSold'])

dtrain.boxplot(column=['YrSold'])  # no outliers
# Some of the non-numeric predictors are stored as numbers; we convert them into strings 

dtrain['MSSubClass'] = dtrain['MSSubClass'].apply(str)

dtrain['YrSold'] = dtrain['YrSold'].astype(str)

dtrain['MoSold'] = dtrain['MoSold'].astype(str)



dtest['MSSubClass'] = dtest['MSSubClass'].apply(str)

dtest['YrSold'] = dtest['YrSold'].astype(str)

dtest['MoSold'] = dtest['MoSold'].astype(str)