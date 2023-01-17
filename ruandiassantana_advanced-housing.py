# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df2 = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

df3 = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
df.head()
df.info()
df.drop(['Street','Alley'],axis=1,inplace=True)

df2.drop(['Street','Alley'],axis=1,inplace=True)
df.info()
df['LotShape'].value_counts()
LotShape = pd.get_dummies(df['LotShape'],drop_first=True)

LotShape2 = pd.get_dummies(df2['LotShape'],drop_first=True)
df = pd.concat((df,LotShape),axis=1)

df2 = pd.concat((df2,LotShape2),axis=1)
df.drop('LotShape',axis=1,inplace=True)

df2.drop('LotShape',axis=1,inplace=True)
MSZoning = pd.get_dummies(df['MSZoning'],drop_first=True)

df = pd.concat((df,MSZoning),axis=1)

df.drop('MSZoning',axis=1,inplace=True)



MSZoning2 = pd.get_dummies(df2['MSZoning'],drop_first=True)

df2 = pd.concat((df2,MSZoning2),axis=1)

df2.drop('MSZoning',axis=1,inplace=True)
LandContour = pd.get_dummies(df['LandContour'],drop_first=True)

df = pd.concat((df,LandContour),axis=1)

df.drop('LandContour',axis=1,inplace=True)



LandContour2 = pd.get_dummies(df2['LandContour'],drop_first=True)

df2 = pd.concat((df2,LandContour2),axis=1)

df2.drop('LandContour',axis=1,inplace=True)
df['Utilities'].value_counts()
df.drop('Utilities',axis=1,inplace=True)

df2.drop('Utilities',axis=1,inplace=True)
df['LotConfig'].value_counts()
def dummie(df,atr):

    x = pd.get_dummies(df[atr],drop_first=True)

    df = pd.concat((df,x),axis=1)

    df.drop(atr,axis=1,inplace=True)

    return df
df = dummie(df,'LotConfig')

df2 = dummie(df2,'LotConfig')
df.head()
df.select_dtypes(['object']).columns
df.drop('LandSlope',axis=1,inplace=True)

df2.drop('LandSlope',axis=1,inplace=True)
df.select_dtypes(['object']).columns
df['Neighborhood'].value_counts()
df.drop('Neighborhood',axis=1,inplace=True)

df2.drop('Neighborhood',axis=1,inplace=True)
df.select_dtypes(['object']).columns
df.drop(['Condition1','Condition2'],axis=1,inplace=True)

df2.drop(['Condition1','Condition2'],axis=1,inplace=True)
df['BldgType'].value_counts()
sns.barplot(x='BldgType',y='SalePrice',data=df)
df = dummie(df,'BldgType')

df2 = dummie(df2,'BldgType')
df.select_dtypes([object]).columns
df['HouseStyle'].value_counts()
sns.barplot(x='HouseStyle',y='SalePrice',data=df)
df = dummie(df,'HouseStyle')

df2 = dummie(df2,'HouseStyle')
df['RoofStyle'].value_counts()
sns.barplot(x='RoofStyle',y='SalePrice',data=df)
df = dummie(df,'RoofStyle')

df2 = dummie(df2,'RoofStyle')
df.select_dtypes(object).columns
df['RoofMatl'].nunique()
df['RoofMatl'].unique()
df['RoofMatl'].value_counts()
df.drop('RoofMatl',axis=1,inplace=True)

df2.drop('RoofMatl',axis=1,inplace=True)
df['Exterior2nd'].value_counts()
plt.figure(figsize=(12,6))

sns.barplot(x='Exterior2nd',y='SalePrice',data=df)
df = dummie(df,'Exterior1st')

df = dummie(df,'Exterior2nd')



df2 = dummie(df2,'Exterior1st')

df2 = dummie(df2,'Exterior2nd')
df['MasVnrType'].nunique()
df['MasVnrType'].unique()
df['MasVnrType'].value_counts()
sns.barplot(x='MasVnrType',y='SalePrice',data=df)
df = dummie(df,'MasVnrType')

df2 = dummie(df2,'MasVnrType')
df.select_dtypes(object).columns
df['ExterQual'].value_counts()
df['ExterCond'].value_counts()
sns.barplot(x='ExterQual',y='SalePrice',data=df)
sns.barplot(x='ExterCond',y='SalePrice',data=df)
df = dummie(df,'ExterQual')

df = dummie(df,'ExterCond')



df2 = dummie(df2,'ExterQual')

df2 = dummie(df2,'ExterCond')
df.info()
df.select_dtypes(object).columns
df['Foundation'].value_counts()
sns.barplot(x='Foundation',y='SalePrice',data=df)
df = dummie(df,'Foundation')

df2 = dummie(df2,'Foundation')
df['BsmtQual'].value_counts()
sns.barplot(x='BsmtQual',y='SalePrice',data=df)
df.groupby('BsmtQual')['SalePrice'].mean()
df = dummie(df,'BsmtQual')

df2 = dummie(df2,'BsmtQual')
df.info()
df.select_dtypes(object)
df['BsmtCond'].nunique()
df['BsmtCond'].unique()
df['BsmtCond'].value_counts()
df['BsmtExposure'].value_counts()
df.drop('BsmtCond',axis=1,inplace=True)

df2.drop('BsmtCond',axis=1,inplace=True)
sns.barplot(x='BsmtExposure',y='SalePrice',data=df)
df.groupby('BsmtExposure')['SalePrice'].mean()
df = dummie(df,'BsmtExposure')

df2 = dummie(df2,'BsmtExposure')
df.select_dtypes(object).columns
df['BsmtFinType1'].value_counts()
sns.barplot(x='BsmtFinType1',y='SalePrice',data=df)
df.groupby('BsmtFinType1')['SalePrice'].mean()
df = dummie(df,'BsmtFinType1')

df2 = dummie(df2,'BsmtFinType1')
df.select_dtypes(object).columns
df['BsmtFinType2'].nunique()
df['BsmtFinType2'].unique()
df['BsmtFinType2'].value_counts()
df.drop('BsmtFinType2',axis=1,inplace=True)

df2.drop('BsmtFinType2',axis=1,inplace=True)
df['Heating'].nunique()
df['Heating'].value_counts()
df.drop('Heating',axis=1,inplace=True)

df2.drop('Heating',axis=1,inplace=True)
df['HeatingQC'].nunique()
df['HeatingQC'].value_counts()
df.groupby('HeatingQC')['SalePrice'].mean()
df = dummie(df,'HeatingQC')

df2 = dummie(df2,'HeatingQC')
df.select_dtypes(object).columns
df['CentralAir'].nunique()
df['CentralAir'].value_counts()
df.drop('CentralAir',axis=1,inplace=True)

df2.drop('CentralAir',axis=1,inplace=True)
df['Electrical'].nunique()
df['Electrical'].value_counts()
df.drop('Electrical',axis=1,inplace=True)

df2.drop('Electrical',axis=1,inplace=True)
df.info()
df.select_dtypes(object).columns
df['KitchenQual'].nunique()
df['KitchenQual'].value_counts()
df = dummie(df,'KitchenQual')

df2 = dummie(df2,'KitchenQual')
df['Functional'].nunique()
df['Functional'].value_counts()
df.drop('Functional',axis=1,inplace=True)

df2.drop('Functional',axis=1,inplace=True)
df['FireplaceQu'].nunique()
df['FireplaceQu'].value_counts()
df.groupby('FireplaceQu')['SalePrice'].mean()
df = dummie(df,'FireplaceQu')

df2 = dummie(df2,'FireplaceQu')
df.info()
df.select_dtypes(object).columns
df['GarageType'].nunique()
df['GarageType'].value_counts()
df.groupby('GarageType')['SalePrice'].mean()
df = dummie(df,'GarageType')

df2 = dummie(df2,'GarageType')
df['GarageFinish'].nunique()
df['GarageFinish'].value_counts()
sns.barplot(x='GarageFinish',y='SalePrice',data=df)
df.groupby('GarageFinish')['SalePrice'].mean()
df = dummie(df,'GarageFinish')

df2 = dummie(df2,'GarageFinish')
df['GarageQual'].nunique()
df['GarageQual'].value_counts()
df.drop('GarageQual',axis=1,inplace=True)

df2.drop('GarageQual',axis=1,inplace=True)
df['GarageCond'].nunique()
df['GarageCond'].value_counts()
df.drop('GarageCond',axis=1,inplace=True)

df2.drop('GarageCond',axis=1,inplace=True)
df.info()
df.select_dtypes(object).columns
df['PavedDrive'].nunique()
df['PavedDrive'].value_counts()
df.drop('PavedDrive',axis=1,inplace=True)

df2.drop('PavedDrive',axis=1,inplace=True)
df['PoolQC'].nunique()
df['PoolQC'].value_counts()
df.drop('PoolQC',axis=1,inplace=True)

df2.drop('PoolQC',axis=1,inplace=True)
df['Fence'].nunique()
df['Fence'].value_counts()
df.drop('Fence',axis=1,inplace=True)

df2.drop('Fence',axis=1,inplace=True)
df.select_dtypes(object).columns
df['MiscFeature'].nunique()
df['MiscFeature'].value_counts()
df.drop('MiscFeature',axis=1,inplace=True)

df2.drop('MiscFeature',axis=1,inplace=True)
df['SaleType'].nunique()
df['SaleType'].value_counts()
df.groupby('SaleType')['SalePrice'].mean()
df.drop('SaleType',axis=1,inplace=True)

df2.drop('SaleType',axis=1,inplace=True)
df['SaleCondition'].nunique()
df['SaleCondition'].value_counts()
df = dummie(df,'SaleCondition')

df2 = dummie(df2,'SaleCondition')
df.info()
df.isnull().sum().sum()
df.isnull().sum().sort_values()
df['LotFrontage']
df.corr()['LotFrontage'].sort_values(ascending=False)[1:]
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
from sklearn.model_selection import train_test_split
X = df.dropna().drop(['LotFrontage','Id','GarageYrBlt','MasVnrArea'],axis=1)

y = df.dropna()['LotFrontage']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
lr.fit(X_train,y_train)
predictions = lr.predict(X_test)
from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score
print(mean_absolute_error(y_test,predictions))

print(mean_squared_error(y_test,predictions))

print(mean_squared_error(y_test,predictions)**0.5)

print(explained_variance_score(y_test,predictions))
lr.fit(X_test,y_test)
X = df.drop(['LotFrontage','Id','GarageYrBlt','MasVnrArea'],axis=1)
predictions = lr.predict(X)
df['LotFrontage'] = df['Id'].apply(lambda x: predictions[x-1])
df['LotFrontage']
df['LotFrontage'].nunique()
df.isnull().sum().sort_values()
df['GarageYrBlt'].nunique()
df['GarageYrBlt'].value_counts()
df.dropna(inplace=True)
df.isnull().sum().sort_values()
df.info()
df.drop(['2.5Fin','Other'],axis=1,inplace=True)

df = df.loc[:,~df.columns.duplicated()]

df2 = df2.loc[:,~df2.columns.duplicated()]
df.drop('Id',axis=1,inplace=True)
from xgboost import XGBRegressor
X = df.drop('SalePrice',axis=1)

y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
model = XGBRegressor()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
print(mean_absolute_error(y_test,predictions))

print(mean_squared_error(y_test,predictions))

print(mean_squared_error(y_test,predictions)**0.5)

print(explained_variance_score(y_test,predictions))
df.describe()['SalePrice']
plt.scatter(y_test,predictions)

plt.plot(y_test,y_test,'r')
df2.info()
df2.isnull().sum().sort_values()
df2.fillna(df.mean(),inplace=True)
model.fit(X_test,y_test)
df2.drop('Id',axis=1,inplace=True)
df2 = scaler.transform(df2)
pred = model.predict(df2)
df3.head()
df3['SalePrice'] = pred
df3.to_csv('pred.csv',line_terminator='\r\n',index=False)