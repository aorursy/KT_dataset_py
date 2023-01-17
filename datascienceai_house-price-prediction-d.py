# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import os
print(os.listdir("../input"))

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.info()
train.columns.values
train.describe()
train['SalePrice'].describe()
train.dtypes
plt.hist(train['SalePrice'])
plt.boxplot(train['SalePrice'])
sns.distplot(train['SalePrice'])
train['SalePrice'].min()
train['SalePrice'].max()
train['SalePrice'].mean()
train[train['SalePrice']> 200000].count()
train[train['SalePrice']< 200000].count()
plt.scatter(x= train['GrLivArea'], y = train['SalePrice'])
plt.scatter(x= train['TotalBsmtSF'], y = train['SalePrice'])
plt.scatter(x= train['OverallQual'], y = train['SalePrice'])
var = 'OverallQual'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
#boxplot for YearBuilt
var = 'YearBuilt'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(20, 10))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);
# Compute the correlation matrix
corrmat = train.corr()

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(16, 12))

# Draw the heatmap
sns.heatmap(corrmat, vmax=.8, square=True);
k = 10 
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], size = 2.5)
plt.show();
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
train = train.drop((missing_data[missing_data['Total'] > 1]).index,1)
train = train.drop(train.loc[train['Electrical'].isnull()].index)
train.isnull().sum().max()
from sklearn.preprocessing import StandardScaler



scaled_salesprice = StandardScaler().fit_transform(train['SalePrice'][:,np.newaxis]);
low_range = scaled_salesprice[scaled_salesprice[:,0].argsort()][:10]
high_range= scaled_salesprice[scaled_salesprice[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)






