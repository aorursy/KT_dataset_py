#invite people for the Kaggle party

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
#bring in the six packs

df_train = pd.read_csv('../input/train.csv')
#check the decoration

df_train.columns
type(df_train['SalePrice'])
df_train['SalePrice'].describe()
sns.distplot(df_train['SalePrice'])
#skewness and kurtosis

print("Skewness: %f" % df_train['SalePrice'].skew())

print("Kurtosis: %f" % df_train['SalePrice'].kurt())
#scatter plot grlivarea/saleprice

var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#scatter plot totalbsmtsf/saleprice

var = 'TotalBsmtSF'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#box plot overallqual/saleprice

var = 'OverallQual'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000)
var#scatter plot overallqual/saleprice

var = 'OverallQual'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
SP = 'SalePrice'

ymax = 800000



var = 'YearBuilt'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)

f, ax = plt.subplots(figsize=(16,8))

fig = sns.boxplot(x=var, y="SalePrice", data = data)

fig.axis(ymin=0, ymax = ymax)

plt.xticks(rotation=90)
# correlation matrix

corrmat = df_train.corr()

f, ax = plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=0.8, square=True)
#saleprice correlation matrix

k = 10 # number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, 

                 fmt='.2f', annot_kws={'size':10},yticklabels=cols.values,

                xticklabels=cols.values)

plt.show()
sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea',

       'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(df_train[cols], size = 2.5)

plt.show();
#missing data

total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1,keys=['Total','Percent'])

missing_data.head(20)
df_train = df_train.drop((missing_data[missing_data['Total']>1]).index, 1)

df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

df_train.isnull().sum().max()
# standardlising data

saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution: ')

print(low_range)

print('\nouter range (high) of the distribution: ')

print(high_range)
# bivariate analysis saleprice/grlivarea

var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#deleting points

df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]

df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)

df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
#bivariate analysis saleprice/grlivarea

var = 'TotalBsmtSF'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
sns.distplot(df_train['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot = plt)
df_train['SalePrice'] = np.log(df_train['SalePrice'])
#transformed histogram and normal probability plot

sns.distplot(df_train['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)
#histogram and normal probability plot

sns.distplot(df_train['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['GrLivArea'], plot=plt)
#data transformation

df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
#transformed histogram and normal probability plot

sns.distplot(df_train['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['GrLivArea'], plot=plt)
#histogram and normal probability plot

sns.distplot(df_train['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)
#create column for new variable (one is enough because it's a binary categorical feature)

#if area>0 it gets 1, for area==0, it gets 0

df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)

df_train['HasBsmt'] = 0

df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1
#transform data

df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
#histogram and normal probability plot

sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm)

fig = plt.figure()

res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)