import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import mean_absolute_error,accuracy_score,mean_squared_error

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor
train=pd.read_csv('../input/home-data-for-ml-course/train.csv')

test=pd.read_csv('../input/home-data-for-ml-course/test.csv')
train.head()
print("Missing values in train data")

missing=train.isnull().sum()

missing[missing>0]
print("Missing values in test data")

missing=test.isnull().sum()

missing[missing>0]
def return_missing_col(data):

    columns=[col for col in data.columns if data[col].isnull().any()]

    return columns
return_missing_col(train)
return_missing_col(test)
train['LotFrontage'].mean()
train['LotFrontage']=train['LotFrontage'].fillna(train['LotFrontage'].mean())

test['LotFrontage']=test['LotFrontage'].fillna(train['LotFrontage']).mean()
train.drop('Alley',axis=1,inplace=True)

test.drop('Alley',axis=1,inplace=True)
train.drop(columns=['MasVnrType','MasVnrArea',],axis=1,inplace=True)

test.drop(columns=['MasVnrType','MasVnrArea',],axis=1,inplace=True)
# the function takes in a type of Series

def check_object(obj):

    objct=[]

    for x in obj:

        if x=='Na':

            objct.append(0)

        else:

            objct.append(1)

    return objct
# we need to fill the misssing values before calling the funnction

train['BsmtQual']=train['BsmtQual'].fillna('Na')

test['BsmtQual']=test['BsmtQual'].fillna('Na')



train['Basement']=pd.DataFrame(check_object(train['BsmtQual']))



test['Basement']=pd.DataFrame(check_object(test['BsmtQual']))
train.drop(columns=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2','BsmtFinSF1','BsmtUnfSF',

                    'TotalBsmtSF'],axis=1,inplace=True)



test.drop(columns=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2','BsmtFinSF1','BsmtUnfSF',

                   'TotalBsmtSF'],axis=1,inplace=True)
train['Electrical'].value_counts()
# filling missing values with 'Mix'

train['Electrical']=train['Electrical'].fillna('Mix')
train.drop(columns=['FireplaceQu'],axis=1,inplace=True)

test.drop(columns=['FireplaceQu'],axis=1,inplace=True)
train['GarageQual']=train['GarageQual'].fillna('Na')

test['GarageQual']=test['GarageQual'].fillna('Na')



train['garage']=pd.DataFrame(check_object(train['GarageQual']))

test['garage']=pd.DataFrame(check_object(test['GarageQual']))
train.drop(columns=['GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond'],axis=1,inplace=True)

test.drop(columns=['GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond'],axis=1,inplace=True)
train['PoolQC']=train['PoolQC'].fillna('Na')

test['PoolQC']=test['PoolQC'].fillna('Na')



train['pool']=pd.DataFrame(check_object(train['PoolQC']))

test['pool']=pd.DataFrame(check_object(test['PoolQC']))
train.drop(columns=['PoolQC','PoolArea'],axis=1,inplace=True)

test.drop(columns=['PoolQC','PoolArea'],axis=1,inplace=True)
train['Fence']=train['Fence'].fillna('Na')

test['Fence']=test['Fence'].fillna('Na')



train['fence']=pd.DataFrame(check_object(train['Fence']))

test['fence']=pd.DataFrame(check_object(test['Fence']))
train.drop('Fence',axis=1,inplace=True)

test.drop('Fence',axis=1,inplace=True)
train.drop(columns=['MiscFeature','MiscVal','MoSold','YrSold','SaleType','SaleCondition'],axis=1,inplace=True)

test.drop(columns=['MiscFeature','MiscVal','MoSold','YrSold','SaleType','SaleCondition'],axis=1,inplace=True)
print("Columns in the train data with missing values are:",len(return_missing_col(train)))
print("Columns with missing values are:",return_missing_col(test))
train.drop(columns=['MSSubClass'],inplace=True,axis=1)

test.drop(columns=['MSSubClass'],inplace=True,axis=1)
test['MSZoning'].isnull().sum()
test['MSZoning']=test['MSZoning'].fillna('RL')
print(" Number of missing values in Utilities are :",test['Utilities'].isnull().sum())

test['Utilities'].value_counts()
test['Utilities']=test['Utilities'].fillna('AllPub')
test.drop(columns=['Neighborhood','LotConfig','LandSlope','Condition1','Condition2','BldgType','HouseStyle','YearRemodAdd','RoofStyle',

                   'RoofMatl','Exterior1st','Exterior2nd','BsmtFullBath','BsmtHalfBath','KitchenQual','Functional'],axis=1,inplace=True)

train.drop(columns=['Neighborhood','LotConfig','LandSlope','Condition1','Condition2','BldgType','HouseStyle','YearRemodAdd','RoofStyle',

                   'RoofMatl','Exterior1st','Exterior2nd','BsmtFullBath','BsmtHalfBath','KitchenQual','Functional'],axis=1,inplace=True)
print("Columns with missing values are:",len(return_missing_col(test)))
train.columns
test.columns
features=['MSZoning','LotFrontage','LotArea','YearBuilt', 'Street','Utilities', 'OverallQual', 'OverallCond','ExterQual', 'ExterCond',

          'Foundation', 'Heating','FullBath', 'HeatingQC','CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF','GrLivArea','BedroomAbvGr', 'KitchenAbvGr',

       'TotRmsAbvGrd', 'Fireplaces','PavedDrive', 'Basement', 'garage','pool', 'fence']
X=train[features]

y=train.SalePrice
X.head()
y.head()
# creates a list of columns with categorical data

s = (X.dtypes == 'object')

categorical_cols = list(s[s].index)
categorical_cols
Encoder_x=LabelEncoder()

for col in categorical_cols:

    X[col]=Encoder_x.fit_transform(X[col])
X.head()
train_x,val_x,train_y,val_y=train_test_split(X,y,train_size=0.8,test_size=0.2)
test2=test[features]
Encoder=LabelEncoder()

for col in categorical_cols:

    test2[col]=Encoder.fit_transform(test2[col])
model=RandomForestRegressor()

model.fit(train_x,train_y)

pred=model.predict(val_x)
print("Mean absolute error:",mean_absolute_error(pred,val_y))

print("Model score",model.score(val_x,val_y))
model=RandomForestRegressor()

model.fit(X,y)

pred=model.predict(test2)
output=pd.DataFrame({'Id':test.Id,'SalePrice':pred})

output.to_csv('submission_rf.csv',index=False)
output.head()
model=XGBRegressor(n_estimators=500,learning_rate=0.05)

model.fit(train_x,train_y,early_stopping_rounds=5,eval_set=[(val_x,val_y)],verbose=False)

pred=model.predict(val_x)

print("Mean absolute error",mean_absolute_error(pred,val_y))

print("Root mean square error",mean_squared_error(pred,val_y,squared=False))
model.fit(X,y,early_stopping_rounds=5,eval_set=[(val_x,val_y)],verbose=False)

pred=model.predict(test2)
output=pd.DataFrame({'Id':test.Id,'SalePrice':pred})

output.to_csv('submission_xgb.csv',index=False)
output.head()