import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')



%matplotlib inline
df_train = pd.read_csv('../input/traindata/train.csv')
df_train.columns
df_train['SalePrice'].describe()
sns.distplot(df_train['SalePrice'])
print("Skewness: %f" % df_train['SalePrice'].skew())

print("Kurtosis: %f" % df_train['SalePrice'].kurt())
data = pd.concat([df_train['SalePrice'], df_train['GrLivArea']], axis=1)

data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,800000))
data = pd.concat([df_train['SalePrice'], df_train['TotalBsmtSF']], axis=1)

data.plot.scatter(x='TotalBsmtSF', y='SalePrice', ylim=(0,800000))
var = 'OverallQual'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(8,6))

fig = sns.boxplot(x=var, y='SalePrice', data=data)

fig.axis(ymin=0, ymax=800000)
var = 'YearBuilt'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x=var, y='SalePrice', data=data)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=60);
cormat = df_train.corr()

f, ax = plt.subplots(figsize = (12, 9))

sns.heatmap(cormat, vmax=.8, square = True)
k=10

cols = cormat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size' : 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
sns.set()

cols = ['SalePrice','OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','YearBuilt']

sns.pairplot(df_train[cols], size=2.5)

plt.show()
total = df_train.isnull().sum().sort_values(ascending=False)

perct = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, perct],axis = 1, keys=['Total','Precent'])

missing_data.head(20)
df_train = df_train.drop((missing_data[missing_data['Total']>1]).index,1)

df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

df_train.isnull().sum().max()
salePrice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis])

low_range = salePrice_scaled[salePrice_scaled[:,0].argsort()][:10]

high_range = salePrice_scaled[salePrice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)
var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]],axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000));
df_train.sort_values(by='GrLivArea', ascending=False)[:2]

df_train = df_train.drop(df_train[df_train['Id']==1299].index)

df_train = df_train.drop(df_train[df_train['Id']==524].index)
var = 'TotalBsmtSF'

data = pd.concat([df_train['SalePrice'],df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
sns.distplot(df_train['SalePrice'], fit=norm)

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)
df_train['SalePrice'] = np.log(df_train['SalePrice'])
sns.distplot(df_train['SalePrice'], fit=norm)

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)
sns.distplot(df_train['GrLivArea'], fit = norm)

fig = plt.figure()

res = stats.probplot(df_train['GrLivArea'], plot=plt)
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
sns.distplot(df_train['GrLivArea'], fit = norm)

fig = plt.figure()

res = stats.probplot(df_train['GrLivArea'], plot=plt)
sns.distplot(df_train['TotalBsmtSF'], fit=norm)

fig = plt.figure()

res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)
df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index = df_train.index)

df_train['HasBsmt'] = 0

df_train.loc[df_train['TotalBsmtSF']>0, 'HasBsmt'] = 1
df_train.loc[df_train['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit = norm)

fig = plt.figure()

res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
plt.scatter(df_train['GrLivArea'], df_train['SalePrice'])
plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF']>0]['SalePrice'])
df_train = pd.get_dummies(df_train)