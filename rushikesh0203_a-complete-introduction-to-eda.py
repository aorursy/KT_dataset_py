

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
data = pd.read_csv('../input/train.csv')

data.head(10)
data.columns

data['SalePrice'].describe()



data.SalePrice.hist()

plt.show()
corr = data.corr()

corr[['SalePrice']].sort_values(by='SalePrice',ascending=False).style.background_gradient()
data.SalePrice.skew()

data.SalePrice.kurtosis()

var='SalePrice'

data.plot(kind='scatter',y=var,x='GrLivArea',figsize=(20,10))
data.plot(kind='scatter',y=var,x='TotalBsmtSF',figsize=(20,10))

data[['SalePrice','OverallQual']].boxplot(by='OverallQual',figsize=(20,10),fontsize=15)

data[['SalePrice','YearBuilt']].boxplot(by='YearBuilt',figsize=(30,15),fontsize=15,rot=90)
data[['SalePrice','GarageCars']].boxplot(by='GarageCars',figsize=(20,10),fontsize=15,rot=90)

data[['SalePrice','1stFlrSF']].plot(kind='scatter',x='1stFlrSF',y='SalePrice',figsize=(20,10))

data[['SalePrice','FullBath']].boxplot(by='FullBath',figsize=(20,10))

(data.isnull().sum()/data.shape[0]).sort_values(ascending=False)[0:20]

data.drop(columns=['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','LotFrontage'],inplace=True)

df=data.isnull().sum().sort_values(ascending=False)

df=pd.concat([df,df/data.shape[0]],axis=1,keys=['Total','Percent']).head(15)

df
data.drop((df[df['Total']>1]).index,1,inplace=True)

data.drop((data.loc[data.Electrical.isnull()]).index,inplace=True)

(data.isnull().sum()/data.shape[0]).sort_values(ascending=False)[0:20]

data.plot(kind='scatter',y=var,x='GrLivArea',figsize=(20,10))

df = data.sort_values(by='GrLivArea',ascending=False)[0:2]

df
data.drop(data[data['Id']==524].index,inplace=True)

data.drop(data[data['Id']==524].index,inplace=True)

data.plot(kind='scatter',y=var,x='TotalBsmtSF',figsize=(20,10),grid=True)

data[['GrLivArea']] = np.log(data[['GrLivArea']])

data[['GrLivArea']].hist(bins=20)

data[['SalePrice']] = np.log(data[['SalePrice']])

data[['SalePrice']].hist(bins=20)

data[['SalePrice']].skew()
data[['GrLivArea']].skew()

data[['TotalBsmtSF']].hist(bins=20)

print(data[['TotalBsmtSF']].skew())

data['TotalBsmtSF']=np.log(data[['TotalBsmtSF']]+1)

data[['TotalBsmtSF']].hist(bins=20)

print(data[['TotalBsmtSF']].skew())
data.plot(kind='scatter',y=var,x='GrLivArea')
data.plot(kind='scatter',y=var,x='TotalBsmtSF')

data.plot(kind='scatter',y=var,x='1stFlrSF',logx=True)
