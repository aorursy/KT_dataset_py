# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
data.info();
data.columns
sns.countplot(x='Total_Bathrooms', data=data)
data['SaleCondition'] = np.where(data['SaleCondition']=='Normal','aaa', data['SaleCondition'])
data.isnull().sum().sort_values(ascending=False).head(25)
dim_null = data.isnull().sum()
dim_null1 = dim_null[dim_null>365].sort_values(ascending=False)
plt.figure(figsize=(8,7))
dim_null1.plot.bar()
data1=data.fillna('null')
data1.isnull().sum().sort_values(ascending=False).head(25)
plt.ylim(0,500000)
sns.boxplot(x='FireplaceQu', y='SalePrice', data=data1)
data1['PoolQC'].value_counts()/data1['PoolQC'].value_counts().sum()
#data1.drop(['MiscFeature','Alley','Fence'], axis=1)
data1.shape

num_dim = data.select_dtypes(exclude='object')
num_dim.columns
num_dim = num_dim.drop('Id',axis=1)
num_dim.hist(figsize=(16, 20), bins=20, xlabelsize=8, ylabelsize=8);
sns.distplot(num_dim['SalePrice'])
num_dim['SalePrice'].mean() + 3*num_dim['SalePrice'].std()
num_dim_clean = num_dim[num_dim['SalePrice']<419248]
corr=num_dim.corr()
corr['SalePrice'].sort_values()
plt.figure(figsize=(10,10))
sns.scatterplot(num_dim_clean['GrLivArea'], num_dim_clean['SalePrice'])
sns.heatmap(num_dim.corr(), vmax=0.8)
sns.pairplot(num_dim_clean[['SalePrice','OverallQual','TotalBsmtSF','GrLivArea','GarageArea','FullBath','YearBuilt','YearRemodAdd']])
sns.boxplot(x='SaleCondition', y='SalePrice', data=data)
cat_dim = data.select_dtypes(include='object')
cat_dim = pd.concat([cat_dim, data['SalePrice']])
corr = data.corr()
corr.shape
data1_tran = StandardScaler().fit_transform(data1['SalePrice'][:,np.newaxis])
print(data1_tran[data1_tran[:,0].argsort()][0:10])
print(data1_tran[data1_tran[:,0].argsort()][-10:])
data1.sort_values(by='GrLivArea', ascending=False)[['Id','GrLivArea','SalePrice']][0:10]
data1.shape
data1=data1.drop(data1[data1['Id']==1299].index);
data1=data1.drop(data1[data1['Id']==524].index);
data1.shape
data2 = pd.get_dummies(data1)
data2.shape