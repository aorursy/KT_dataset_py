# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import seaborn as sns
import os
import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.
train_data=pd.read_csv('../input/train.csv')
test_data=pd.read_csv('../input/test.csv')
train_data.head()

train_data.describe()

train_data.info()
null_columns=train_data.columns[train_data.isnull().any()]
null_columns_sum=train_data[null_columns].isnull().sum()
print(null_columns)

print(null_columns_sum)
percentage_Null=null_columns_sum/train_data.shape[0]
print(percentage_Null)
train_data.drop(columns=['Alley','PoolQC','Fence','MiscFeature'],axis=1, inplace=True )
train_data.drop(columns=['Id','GarageYrBlt'],axis =1 ,inplace=True)
train_data.dropna(axis=0,inplace=True)
train_data.head()
corrmat=train_data.corr()
plt.subplots(figsize=(15,12))
sns.heatmap(corrmat,vmax=.8)

train_data['LotFrontage']=train_data['LotFrontage'].fillna(value=train_data['LotFrontage'].mean())
train_data['FireplaceQu'] = train_data['FireplaceQu'].fillna(value='No_FP')
train_data['BsmtCond'] =  train_data['BsmtCond'].fillna(value='No_BS')
train_data['BsmtQual'] =  train_data['BsmtQual'].fillna(value='No_BS')
for col in ('BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    train_data[col] = train_data[col].fillna('No_BS')
for col in ('GarageQual', 'GarageCond', 'GarageFinish','GarageType'):
    train_data[col] = train_data[col].fillna('No_Garage')
    

train_data = train_data[['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
       'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',
       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',
       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
       'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC',
       'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
       'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageFinish',
       'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive',
       'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
       'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition','SalePrice']]
from sklearn.preprocessing import LabelEncoder
labelencoder= LabelEncoder()
for col in train_data.columns.values:
    if train_data[col].dtype=='object':
        labelencoder.fit(train_data[col].values)
        train_data[col]= labelencoder.transform(train_data[col])
'''
pd.get_dummies(train_data,columns=['MSZoning', 'Street','LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope','Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle',
       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType','ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
       'BsmtExposure', 'BsmtFinType1','BsmtFinType2','Heating', 'HeatingQC',
       'CentralAir', 'Electrical','KitchenQual', 'TotRmsAbvGrd',
       'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageFinish','GarageQual', 'GarageCond', 'PavedDrive','SaleType',
       'SaleCondition'],drop_first=True)
'''
sns.distplot(train_data['SalePrice'])
X = train_data.iloc[:,0:74]
Y = train_data.iloc[:,-1]
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=500, random_state=2, oob_score=True)
regressor.fit(X,Y)
print('Out-of-bag score estimate:', (regressor.oob_score_))
test_data.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)
test_data.drop(['Id', 'GarageYrBlt'], axis=1,inplace=True)
test_data['LotFrontage']=test_data['LotFrontage'].fillna(value=train_data['LotFrontage'].mean())
test_data['FireplaceQu'] = test_data['FireplaceQu'].fillna(value='No_FP')
test_data['BsmtCond'] =  test_data['BsmtCond'].fillna(value='No_BS')
test_data['BsmtQual'] =  test_data['BsmtQual'].fillna(value='No_BS')
for col in ('BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    test_data[col] = test_data[col].fillna('No_BS')
for col in ('GarageQual', 'GarageCond', 'GarageFinish','GarageType'):
    test_data[col] = test_data[col].fillna('No_Garage')
    
test_data.dropna(axis=0, inplace=True)
for col in test_data.columns.values:
       # Encoding only categorical variables
        if test_data[col].dtypes=='object':
            labelencoder.fit(test_data[col].values)
            test_data[col]=labelencoder.transform(test_data[col])
'''
pd.get_dummies(test_data,columns=['MSZoning', 'Street','LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope','Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle',
       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType','ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
       'BsmtExposure', 'BsmtFinType1','BsmtFinType2','Heating', 'HeatingQC',
       'CentralAir', 'Electrical','KitchenQual', 'TotRmsAbvGrd',
       'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageFinish','GarageQual', 'GarageCond', 'PavedDrive','SaleType',
       'SaleCondition'],drop_first=True)
'''
X_test = test_data.iloc[:,0:74]
Y_test = test_data.iloc[:,-1]
y_pred = regressor.predict(X_test)
print(y_pred)
my_submission = pd.DataFrame({'Id': Y_test, 'SalePrice': y_pred})
my_submission.to_csv('submission.csv', index=False)
''''
ss = pd.read_csv('submission.csv')
ss.loc[:, 'SalePrice'] = y_pred
ss.to_csv('sub.csv',
          index=False)
          '''

