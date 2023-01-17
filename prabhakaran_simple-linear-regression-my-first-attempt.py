
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
train_no_SalePrice=train.loc[:,train.columns!='SalePrice']
y=train['SalePrice']
train_test=pd.concat([train_no_SalePrice,test])
# Lets replace NAN in PoolQC to 'Fa' since only 3 values are missing when PoolArea>0 also probably there is no Pool
train_test.loc[(train_test['PoolArea']>0) & (train_test['PoolQC'].isnull()),['PoolQC']]='Fa'
# Since remaining PoolQC is null because no pools in the house , replace nulls with 'None'
train_test.loc[(train_test['PoolQC'].isnull()),['PoolQC']]='None'
# Probably there is no MiscFeature , replace nulls with 'None'
train_test.loc[(train_test['MiscFeature'].isnull()),['MiscFeature']]='None'
#Lets fill the null with NA for below 3 features
train_test['Alley']=train_test['Alley'].fillna('None')
train_test['Fence']=train_test['Fence'].fillna('None')
train_test['FireplaceQu']=train_test['FireplaceQu'].fillna('None')
# LotFrontage: Linear feet of street connected to property
# Hence lets take the median of the property connected to the street by Neighborhood
train_test['LotFrontage']=train_test.groupby('Neighborhood')['LotFrontage'].transform(lambda x:x.fillna(x.median()))
for col in ['GarageCond','GarageType','GarageFinish','GarageQual']:
    train_test[col]=train_test[col].fillna('None')
for col in ['GarageYrBlt','GarageCars','GarageArea']:
    train_test[col]=train_test[col].fillna(0)
for col in ['BsmtExposure','BsmtCond','BsmtQual','BsmtFinType2','BsmtFinType1']:
    train_test[col]=train_test[col].fillna('None')
for col in ['MasVnrArea']:
    train_test[col]=train_test[col].fillna(0)
for col in ['MasVnrType']:
    train_test[col]=train_test[col].fillna('None')
# Since only 4 NULLS , replcae it with Mode 
# train_test.MSZoning.mode()[0] = 'RL'
train_test['MSZoning']=train_test['MSZoning'].fillna('RL')
# Replace Nulls with Mode
# train_test.Utilities.value_counts()
# AllPub    2916
# NoSeWa       1
# Name: Utilities, dtype: int64
#

train_test['Utilities']=train_test['Utilities'].fillna('AllPub')
# Replace Nulls with Mode
# train_test.Functional.value_counts()
# Typ     2717
# Min2      70
# Min1      65
# Mod       35
# Maj1      19
# Maj2       9
# Sev        2
# Name: Functional, dtype: int64

train_test['Functional']=train_test['Functional'].fillna('Typ')
# Replace all the null values with mode
for col in ['BsmtFullBath','BsmtHalfBath','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','SaleType','BsmtFinSF1','Exterior1st','Electrical',\
            'Exterior2nd','KitchenQual']:
    fill_value=train_test[col].mode()[0]
    train_test[col]=train_test[col].fillna(fill_value)
sns.heatmap(train_test.isnull())
#droping Id column in train_test as it is not useful
train_test.drop(columns=['Id'],inplace=True)
dummy_train_test=pd.get_dummies(train_test)
dummy_train=dummy_train_test[:train.shape[0]]
dummy_test=dummy_train_test[train.shape[0]:]
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
#RMSE metric
from sklearn.metrics import mean_squared_error
lr=LinearRegression()
lr.fit(dummy_train,y)

import random
nrow=dummy_train.shape[0]
rmse=[]
for i in range(1000):
    random_rows=random.sample(range(nrow),int(nrow*.15))
    rmse.append(np.sqrt(mean_squared_error(lr.predict(dummy_train.iloc[random_rows]),y[random_rows])))
sns.distplot(rmse)
#Mean and Standard Deviation
[np.std(rmse),np.mean(rmse)]




