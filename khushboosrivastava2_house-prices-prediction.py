import gc

gc.collect()

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
train = pd.read_csv('../input/train.csv')

train.columns

test = pd.read_csv('../input/test.csv')

test.columns

train['SalePrice'].describe()
sns.distplot(train['SalePrice'])
#skewness and kurtosis

print("Skewness: %f" % train['SalePrice'].skew())

print("Kurtosis: %f" % train['SalePrice'].kurt())
var = '1stFlrSF'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
var = 'GrLivArea'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
var = 'OverallCond'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000)
var = 'OverallQual'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000)
var = 'YearBuilt'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(20, 10))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=90)
#correlation matrix

cm = train.corr()

f, ax = plt.subplots(figsize=(15, 10))

sns.heatmap(cm, vmax=1, square=True)

#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = cm.nlargest(k, 'SalePrice')['SalePrice'].index

cmat = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cmat, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#scatterplot

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', '1stFlrSF', 'FullBath', 'YearBuilt']

sns.pairplot(train[cols], size = 2.5)

plt.show()
#missing data

total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
#Dealing with missing data

train = train.drop((missing_data[missing_data['Total'] > 1]).index,1)

train = train.drop(train.loc[train['Electrical'].isnull()].index) 

#In summary, to handle missing data, we'll delete all the variables with missing data, except the variable 'Electrical'. In 'Electrical' we'll just delete the observation with missing data.

train.isnull().sum().max()
#standardizing data

saleprice_scaled = StandardScaler().fit_transform(train['SalePrice'][:,np.newaxis]);

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)
#deleting points

train.sort_values(by = 'GrLivArea', ascending = False)[:2]

train = train.drop(train[train['Id'] == 1299].index)

train = train.drop(train[train['Id'] == 524].index)
#histogram and normal probability plot

sns.distplot(train['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)
#applying log transformation

train['SalePrice'] = np.log(train['SalePrice'])
#transformed histogram and normal probability plot

sns.distplot(train['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)
#histogram and normal probability plot

sns.distplot(train['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(train['GrLivArea'], plot=plt)
#data transformation

train['GrLivArea'] = np.log(train['GrLivArea'])
#transformed histogram and normal probability plot

sns.distplot(train['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(train['GrLivArea'], plot=plt)
#histogram and normal probability plot

sns.distplot(train['1stFlrSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(train['1stFlrSF'], plot=plt)
#data transformation

train['1stFlrSF'] = np.log(train['1stFlrSF'])
#transformed histogram and normal probability plot

sns.distplot(train['1stFlrSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(train['1stFlrSF'], plot=plt)
#create column for new variable

#if area>0 it gets 1, for area==0 it gets 0

train['Has1stFlr'] = pd.Series(len(train['1stFlrSF']), index=train.index)

train['Has1stFlr'] = 0 

train.loc[train['1stFlrSF']>0,'Has1stFlr'] = 1

#transform data

train.loc[train['Has1stFlr']==1,'1stFlrSF'] = np.log(train['1stFlrSF'])
#histogram and normal probability plot

sns.distplot(train[train['1stFlrSF']>0]['1stFlrSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(train[train['1stFlrSF']>0]['1stFlrSF'], plot=plt)

#scatter plot

plt.scatter(train['GrLivArea'], train['SalePrice'])
#scatter plot

plt.scatter(train[train['1stFlrSF']>0]['1stFlrSF'], train[train['1stFlrSF']>0]['SalePrice'])
#convert categorical variable into dummy

train = pd.get_dummies(train)