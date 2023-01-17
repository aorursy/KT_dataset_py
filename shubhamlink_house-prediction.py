

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_c
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy import stats
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_train=pd.read_csv("../input/train.csv")
df_train.columns

df_train['SalePrice'].describe()
sns.distplot(df_train['SalePrice'])
data=pd.concat([df_train['SalePrice'],df_train['GrLivArea']],axis=1)
data.plot.scatter(x='GrLivArea',y='SalePrice',ylim=(0,800000))
data=pd.concat([df_train['SalePrice'],df_train['TotalBsmtSF']],axis=1)
data.plot.scatter(x='TotalBsmtSF',y='SalePrice',ylim=(0,800000))
cormatrix=df_train.corr()
f,ax=plt.subplots(figsize=(12,9))
sns.heatmap(cormatrix,vmax=0.8,square=True)
sns.set()
cols=['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols],size=2.5)
t_missing=df_train.isnull().sum().sort_values(ascending=False)
percent=(df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
data_missing=pd.concat([t_missing,percent],axis=1,keys=['No of missing values','Percent'])
data_missing.head(20)
df_train=df_train.drop(data_missing[data_missing['Percent']>0.004].index,1)
df_train=df_train.drop(data_missing.loc[df_train['Electrical'].isnull()].index)

df_train = df_train.drop((data_missing[data_missing['No of missing values'] > 1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
df_train.isnull().sum().max()
saleprice_scaled=StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);
low_range=saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range=saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
print(low_range)
print(high_range)

data=pd.concat([df_train["SalePrice"],df_train['GrLivArea']],axis=1)
data.plot.scatter(x='GrLivArea',y='SalePrice',ylim=(0,800000))
df_train.sort_values(by='GrLivArea',ascending=False)[:2]
df_train=df_train.drop(df_train[df_train['Id']==1299].index)
df_train=df_train.drop(df_train[df_train['Id']==524].index)
sns.distplot(df_train['SalePrice']);
fig=plt.figure()
res=stats.probplot(df_train['SalePrice'],plot=plt)

df_train['SalePrice']=np.log(df_train['SalePrice'])
sns.distplot(df_train['SalePrice']);
fig=plt.figure()
res=stats.probplot(df_train['SalePrice'],plot=plt)
sns.distplot(df_train['GrLivArea']);
fig=plt.figure()
res=stats.probplot(df_train['GrLivArea'],plot=plt)
df_train['GrLivArea']=np.log(df_train['GrLivArea'])
sns.distplot(df_train['GrLivArea']);
fig=plt.figure()
res=stats.probplot(df_train['GrLivArea'],plot=plt)
sns.distplot(df_train['TotalBsmtSF']);
fig=plt.figure()
res=stats.probplot(df_train['TotalBsmtSF'],plot=plt)
df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0 
df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1
df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF']);
fig = plt.figure()
res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
plt.scatter(df_train['GrLivArea'], df_train['SalePrice']);
plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF']>0]['SalePrice']);