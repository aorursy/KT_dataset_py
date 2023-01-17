# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train = pd.read_csv('../input/house-prices-data/train.csv')

df_train
df_train.columns
df_train['SalePrice'].describe()
#Histogram

sns.distplot(df_train['SalePrice'])
#skewness and kurtosis

print('skewness: %f' % df_train['SalePrice'].skew())

print('kurtosis: %f' % df_train['SalePrice'].kurt())
#Scattor Plot

var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'],df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
#Scattor Plot

var = 'TotalBsmtSF'

data = pd.concat([df_train['SalePrice'],df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
var = 'OverallQual'

data = pd.concat([df_train['SalePrice'],df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(8,6))

fig= sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000)
var = 'YearBuilt'

data = pd.concat([df_train['SalePrice'],df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(16,8))

fig= sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000)

plt.xticks(rotation=90)
#Pair Plot

sns.set()

cols=['SalePrice','OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','YearBuilt']

sns.pairplot(df_train[cols], size=2.5)

plt.show()
# Corelation Matrix



corrmat= df_train.corr()

print(corrmat)
corrmat= df_train.corr()

f, ax = plt.subplots(figsize =(12,9))

sns.heatmap(corrmat,vmax=8, square = True)
# SalePrice Correlation Matrix



k=10 #number of variables in the heat map

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm= np.corrcoef(df_train[cols].values.T)

sns.set(font_scale =1.25)

hm= sns.heatmap(cm,cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size':10},

               yticklabels=cols.values, xticklabels=cols.values)



plt.show()
#missing Data

total = df_train.isnull().sum().sort_values(ascending=False)

percent= (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total,percent], axis=1, keys=['Total','Percent'])

missing_data.head(20)
#df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index, 1)

df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)

df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

df_train.isnull().sum().max()

#standardizing data

saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)