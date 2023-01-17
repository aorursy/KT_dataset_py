# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt 

import seaborn as sns 

sns.set()

from scipy.stats import norm 

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings 

warnings.filterwarnings('ignore')

%matplotlib inline
df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df_train.columns
df_train.shape
df_train.describe()
df_train.SalePrice.describe()
sns.distplot(df_train['SalePrice'])

#Deviate from the normal distribution.

#Have appreciable positive skewness.

#Show peakedness.
#skewness and kurtosis

print("Skewness: %f" % df_train['SalePrice'].skew())

print("Kurtosis: %f" % df_train['SalePrice'].kurt())
var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x = var, y = 'SalePrice', ylim=(0,800000))

#Linear Relationship
#scatter plot totalbsmtsf/saleprice

var = 'TotalBsmtSF'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))

#Strong LInear or Exponential 
var = 'OverallQual'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(8,6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000)
var = 'YearBuilt'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(20, 10))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=90);

#SalePrice' is more prone to spend more money in new stuff than in old relics.
#saleprice correlation matrix

corrmat = df_train.corr()

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()

#scatterplot

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(df_train[cols], size = 2.5)

plt.show();
#Finding NULL data 

total = df_train.isnull().sum().sort_values(ascending = False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
#dealing with missing data

df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)

df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

df_train.isnull().sum().max() #just checking that there's no missing data missing...
#Standardizing Data 

saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis])

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)
#Histogram and normal distribution

sns.distplot(df_train['SalePrice'], fit = norm)

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot = plt)
#Histogram and normal distribution

sns.distplot(df_train['SalePrice'], fit = norm)

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot = plt)
#Applying log transformations

df_train['SalePrice'] = np.log(df_train['SalePrice'])
#Transformed Histogram and normal probability plot

sns.distplot(df_train['SalePrice'], fit = norm)

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot = plt)
#histogram and normal probability plot

sns.distplot(df_train['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)
#create column for new variable (one is enough because it's a binary categorical feature)

#if area>0 it gets 1, for area==0 it gets 0



df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)

df_train['HasBsmt'] = 0 

df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1
#transform data

df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
#histogram and normal probability plot

sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
#scatter plot

plt.scatter(df_train['GrLivArea'], df_train['SalePrice']);
#scatter plot

plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF']>0]['SalePrice']);
#convert categorical variable into dummy

df_train = pd.get_dummies(df_train)

df_train.head()
df_train.to_csv('House_Price_Prediction_Sid.csv')