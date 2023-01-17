import numpy as np
import pandas as pd
pd.set_option("display.max_column",80)
pd.set_option("display.max_row",1000)
pd.set_option('display.width', 1000)
import matplotlib as plt
data=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
data
data.isnull().sum()

data.corr()
data.columns
df=data.drop(['PoolQC','Utilities','Fence','MiscFeature','Alley','Id','GarageYrBlt','MSSubClass','PoolArea','MiscVal','MoSold','YrSold','Street','LotConfig','LandSlope','Condition2','EnclosedPorch','KitchenAbvGr','BedroomAbvGr','BsmtHalfBath','HalfBath','BsmtFinType2', 'BsmtFinSF2','LowQualFinSF','BsmtUnfSF','Exterior2nd','Heating'],axis=1)
df
df.dtypes
df['FireplaceQu']=df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])
df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])
df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])
df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
df['BsmtFinType1']=df['BsmtFinType1'].fillna(df['BsmtFinType1'].mode()[0])
df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])
df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
df
df.isnull().sum()

from sklearn.impute import SimpleImputer
imp=SimpleImputer(copy=True, missing_values=np.nan, strategy='mean', verbose=0)
imp
df.iloc[:,1:2]=imp.fit_transform(df.iloc[:,1:2])
df
df.iloc[:,17:18]=imp.fit_transform(df.iloc[:,17:18])
df
df.isnull().sum()
df.dropna(inplace=True)

df
X=df.iloc[:,:-1]
X
y=df.iloc[:,-1]
y
from sklearn.preprocessing import LabelEncoder
categorical_feature_mask = X.dtypes==object    #filters categorical features using boolean mask
categorical_cols = X.columns[categorical_feature_mask].tolist()  #filters categorical columns using mask and turns into list
categorical_cols
lb=LabelEncoder()
X[categorical_cols] = X[categorical_cols].apply(lambda x: lb.fit_transform(x))
X
df=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
df
df.isnull().sum()
df.corr()
Xtest=df.drop(['PoolQC','Utilities','Fence','MiscFeature','Alley','Id','GarageYrBlt','MSSubClass','PoolArea','MiscVal','MoSold','YrSold','Street','LotConfig','LandSlope','Condition2','EnclosedPorch','KitchenAbvGr','BedroomAbvGr','BsmtHalfBath','HalfBath','BsmtFinType2', 'BsmtFinSF2','LowQualFinSF','BsmtUnfSF','Exterior2nd','Heating'],axis=1)
Xtest
Xtest.dtypes
Xtest['MSZoning']=Xtest['MSZoning'].fillna(Xtest['MSZoning'].mode()[0])
Xtest['Exterior1st']=Xtest['Exterior1st'].fillna(Xtest['Exterior1st'].mode()[0])
Xtest['MasVnrType']=Xtest['MasVnrType'].fillna(Xtest['MasVnrType'].mode()[0])
Xtest['BsmtQual']=Xtest['BsmtQual'].fillna(Xtest['BsmtQual'].mode()[0])
Xtest['BsmtCond']=Xtest['BsmtCond'].fillna(Xtest['BsmtCond'].mode()[0])
Xtest['BsmtExposure']=Xtest['BsmtExposure'].fillna(Xtest['BsmtExposure'].mode()[0])
Xtest['BsmtFinType1']=Xtest['BsmtFinType1'].fillna(Xtest['BsmtFinType1'].mode()[0])
Xtest['KitchenQual']=Xtest['KitchenQual'].fillna(Xtest['KitchenQual'].mode()[0])
Xtest['Functional']=Xtest['Functional'].fillna(Xtest['Functional'].mode()[0])
Xtest['FireplaceQu']=Xtest['FireplaceQu'].fillna(Xtest['FireplaceQu'].mode()[0])
Xtest['GarageType']=Xtest['GarageType'].fillna(Xtest['GarageType'].mode()[0])
Xtest['GarageFinish']=Xtest['GarageFinish'].fillna(Xtest['GarageFinish'].mode()[0])
Xtest['GarageQual']=Xtest['GarageQual'].fillna(Xtest['GarageQual'].mode()[0])
Xtest['GarageCond']=Xtest['GarageCond'].fillna(Xtest['GarageCond'].mode()[0])
Xtest['SaleType']=Xtest['SaleType'].fillna(Xtest['SaleType'].mode()[0])
Xtest
Xtest.isnull().sum()
from sklearn.impute import SimpleImputer
imp1=SimpleImputer(missing_values=np.nan, strategy='mean')
imp1
Xtest.iloc[:,1:2]=imp.fit_transform(Xtest.iloc[:,1:2])
Xtest
Xtest.iloc[:,17:18]=imp.fit_transform(Xtest.iloc[:,17:18])
Xtest
Xtest.iloc[:,25:27]=imp.fit_transform(Xtest.iloc[:,25:27])
Xtest
Xtest.iloc[:,33:34]=imp.fit_transform(Xtest.iloc[:,33:34])
Xtest
Xtest.iloc[:,42:44]=imp.fit_transform(Xtest.iloc[:,42:44])
Xtest
Xtest.isnull().sum()
from sklearn.preprocessing import LabelEncoder
categorical_feature_mask = Xtest.dtypes==object    #filters categorical features using boolean mask
categorical_cols = Xtest.columns[categorical_feature_mask].tolist()
categorical_cols
lb1=LabelEncoder()
Xtest[categorical_cols] = Xtest[categorical_cols].apply(lambda x: lb1.fit_transform(x))
Xtest
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
bbc=BaggingRegressor(n_estimators=70,random_state=42,base_estimator=LinearRegression())
model=bbc.fit(X,y)
ypred2=model.predict(Xtest)
ypred2
yp=pd.DataFrame(ypred2,columns=['SalePrice'])
yp
e=pd.DataFrame(df['Id'])
e
submission=pd.concat([e,yp],axis=1)
submission.to_csv('submission.csv')
