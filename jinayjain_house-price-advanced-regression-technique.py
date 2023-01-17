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

pd.set_option('display.float_format',lambda x: '{:.3f}'.format(x))
train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.head()
test.head()
print(train.shape)
print(test.shape)
train.describe()
test.head()
sns.distplot(train['SalePrice'])
feature=train.select_dtypes(include=[np.int,np.float])
feature.columns
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
sns.scatterplot(train['SalePrice'],train['GrLivArea'])
sns.barplot(x='YrSold',y='SalePrice',data=train)
sns.scatterplot(train['YearBuilt'],train['SalePrice'])
sns.heatmap(train.isnull(),cbar=False)
train['PoolQC']=train['PoolQC'].fillna('None')
train['MiscFeature']=train['MiscFeature'].fillna('None')
train['Alley']=train['Alley'].fillna('None')
train['Fence']=train['Fence'].fillna('None')
train['FireplaceQu']=train['FireplaceQu'].fillna('None')
for col in ('GarageCond','GarageQual','GarageFinish','GarageYrBlt','GarageType'):
    train[col]=train[col].fillna('None')
for col in ('BsmtFinType2','BsmtExposure','BsmtFinType1','BsmtCond','BsmtQual'):
    train[col]=train[col].fillna('None')
train['MasVnrArea']=train['MasVnrArea'].fillna('0')
train['MasVnrType']=train['MasVnrType'].fillna('None')
train['LotFrontage']=train['LotFrontage'].fillna(0)
train['Electrical']=train['Electrical'].fillna(0)
train_na = (train.isnull().sum() / len(train)) * 100
train_na = train_na.drop(train_na[train_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :train_na})
missing_data.head(20)
test_na = (test.isnull().sum() / len(test)) * 100
test_na = train_na.drop(train_na[train_na == 0].index).sort_values(ascending=False)[:30]
missing_data1 = pd.DataFrame({'Missing Ratio' :test_na})
missing_data1.head(20)
objList = train.select_dtypes(include = "object").columns
print (objList)
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()

for col1 in objList:
    train[col1] = label.fit_transform(train[col1].astype(str))

print (train.info())
objList1 = test.select_dtypes(include = "object").columns
print (objList1)
from sklearn.preprocessing import LabelEncoder
label1 = LabelEncoder()

for col2 in objList:
    test[col2] = label1.fit_transform(test[col2].astype(str))

print (test.info())
x=train.drop('SalePrice',axis=1)
y=train['SalePrice']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor(n_estimators=10)
model.fit(x_train,y_train)
y_predict=model.predict(x_test)
print(y_predict)
from sklearn.metrics import mean_squared_error,r2_score
mse=mean_squared_error(y_test,y_predict)
print('RMSE',np.sqrt(mse))
print('Variance score: %.2f' % r2_score(y_test, y_predict))

print("Result :",model.score(x_train, y_train))