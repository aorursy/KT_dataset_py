import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
df_train = pd.read_csv('../input/train.csv')
df_train.columns
df_train['SalePrice'].describe()
#histogram

sns.distplot(df_train['SalePrice'])
print("Skewness of Sale Price: " ,df_train['SalePrice'].skew())

print("Kurtosis of Sale Price: " ,df_train['SalePrice'].kurt())
df_train['SalePrice_log'] = np.log(df_train['SalePrice'])
sns.distplot(df_train['SalePrice_log'])
print("Skewness of Sale Price Log: " ,df_train['SalePrice_log'].skew())

print("Kurtosis of Sale Price Log: " ,df_train['SalePrice_log'].kurt())
#scatter plot grlivarea/saleprice

var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
df_train[df_train['GrLivArea'] > 4000]
id_to_del = df_train.index[df_train['GrLivArea'] > 4000].tolist()
df_train.drop(df_train.index[id_to_del], inplace=True)
df_train[df_train['GrLivArea'] > 4000]
#scatter plot grlivarea/saleprice

var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#scatter plot totalbsmtsf/saleprice

var = 'TotalBsmtSF'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
id_to_del = df_train.index[df_train['TotalBsmtSF'] > 4000].tolist()
df_train.drop(df_train.index[id_to_del], inplace=True)
#scatter plot totalbsmtsf/saleprice

var = 'TotalBsmtSF'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#box plot overallqual/saleprice

var = 'OverallQual'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000)
var = 'YearBuilt'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=90);
#correlation matrix

corrmat = df_train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
print(df_train.corr())
#scatter plot totalbsmtsf/saleprice

var = 'LotArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
df_train['Total_Area'] = np.sum(df_train[['1stFlrSF' ,'2ndFlrSF' ,'GrLivArea','GarageArea','OpenPorchSF','PoolArea','WoodDeckSF','EnclosedPorch','3SsnPorch','ScreenPorch']],axis=1)
#scatter plot totalbsmtsf/saleprice

var = 'Total_Area'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
df_train = pd.get_dummies(df_train)
df_train.head()
df_train['Total_Area'].cat.categories