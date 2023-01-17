# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from scipy.stats import pearsonr,norm, probplot

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
train_df.columns
train_df['SalePrice'].describe()
#histogram

sns.distplot(train_df['SalePrice'])
print("Skewness: %f" % train_df['SalePrice'].skew())

print("Kurtosis: %f" % train_df['SalePrice'].kurt())
#Scatter plot grlivarea/Saleprice

var = 'GrLivArea'

data = pd.concat([train_df['SalePrice'],train_df[var]],axis=1)

data.plot.scatter(x=var,y='SalePrice',ylim=(0,800000))
gLivArea_SalePrice_corr = pearsonr(train_df['SalePrice'],

                                  train_df[var])

gLivArea_SalePrice_corr
# scatter plot totalbsmtsf/saleprice

var = 'TotalBsmtSF'

data = pd.concat([train_df['SalePrice'],train_df[var]],axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
totalBsmSF_SalePrice_corr = pearsonr(train_df['SalePrice'],

                                    train_df[var])

totalBsmSF_SalePrice_corr
#box plot overallqual/saleprice

var = 'OverallQual'

data = pd.concat([train_df['SalePrice'], train_df[var]],axis=1)

f, ax = plt.subplots(figsize=(8,6))

fig = sns.boxplot(x=var, y='SalePrice', data = data)

fig.axis(ymin=0, ymax=800000)
var = 'YearBuilt'

data = pd.concat([train_df['SalePrice'],train_df[var]], axis=1)

f, ax = plt.subplots(figsize=(16,8))

fig = sns.boxplot(x=var, y='SalePrice', data=data)

fig.axis(ymin=0, ymax=800000)

plt.xticks(rotation=90)
#Correlation matrix heat map style

corrmat = train_df.corr()

f,ax = plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=.8,square=True)
# Correlation matrix  zoomed heat map style

k = 10 # No of variables for heat map

cols = corrmat.nlargest(k,'SalePrice')['SalePrice'].index

cm = np.corrcoef(train_df[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size':10},

                 yticklabels=cols.values,xticklabels=cols.values)

plt.show()
sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(train_df[cols],size=2.5)

plt.show()
total = train_df.isnull().sum().sort_values(ascending=False)

percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total,percent],axis=1,keys=['Total','Percent'])

missing_data.head(20)
# deleting with missing data

train_df = train_df.drop(missing_data[missing_data['Total'] > 1].index,1)

train_df = train_df.drop(train_df.loc[train_df['Electrical'].isnull()].index,)

train_df.isnull().sum().max() 
# Univariate analysis of Sale Price

sale_price_scaled = StandardScaler().fit_transform(train_df['SalePrice'][:,np.newaxis])

low_range = sale_price_scaled[sale_price_scaled[:,0].argsort()][:10]

high_range = sale_price_scaled[sale_price_scaled[:,0].argsort()][-10:]

print('Outer range (low) of the distribution: ')

print(low_range)

print('Outer range (high) of the distributions: ')

print(high_range)
var = 'GrLivArea'

data = pd.concat([train_df['SalePrice'], train_df[var]],axis=1)

data.plot.scatter(x=var, y='SalePrice',ylim=(0,800000))
# deleting outlier points

train_df.sort_values(by='GrLivArea', ascending = False)[:2]
train_df = train_df.drop(train_df[train_df['Id']==1299].index)

train_df = train_df.drop(train_df[train_df['Id']==524].index)
var = 'TotalBsmtSF'

data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)

data.plot.scatter(x=var,y='SalePrice', ylim=(0,800000))
#deleting outliers

train_df.sort_values(by='TotalBsmtSF',ascending = False)[:2]
train_df.drop(train_df[train_df['Id']==333].index, inplace=True)

train_df.drop(train_df[train_df['Id']==497].index, inplace=True)
#histogram and normal probablity plot

sns.distplot(train_df['SalePrice'], fit=norm)

fig=plt.figure()

res=probplot(train_df['SalePrice'], plot=plt)
# apply log transformations

train_df['SalePrice'] = np.log(train_df['SalePrice'])
#transformed histogram and normal probability plot 

sns.distplot(train_df['SalePrice'], fit=norm)

fig = plt.figure()

res = probplot(train_df['SalePrice'], plot=plt)
sns.distplot(train_df['GrLivArea'], fit=norm)

fig = plt.figure()

res = probplot(train_df['GrLivArea'], plot=plt)
#Living area transformation

train_df['GrLivArea'] = np.log(train_df['GrLivArea'])
sns.distplot(train_df['GrLivArea'], fit=norm)

fig = plt.figure()

res = probplot(train_df['GrLivArea'], plot=plt)
sns.distplot(train_df['TotalBsmtSF'], fit=norm)

fig = plt.figure()

res = probplot(train_df['TotalBsmtSF'], plot = plt)
# Create a categorical variabe to identify if there is a basement

train_df["HasBsmt"] = pd.Series(len(train_df["TotalBsmtSF"]), index=train_df.index)

train_df['HasBsmt'] = 0

train_df.loc[train_df['TotalBsmtSF']>0,'HasBsmt'] = 1
# Transform data

train_df.loc[train_df['HasBsmt']==1,'TotalBsmtSF'] = np.log(train_df['TotalBsmtSF'])
#histogram and normal probability plot

sns.distplot(train_df[train_df['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm)

fig = plt.figure()

res = probplot(train_df[train_df['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
# Scatter plot between living area and sale price after  transformation

plt.scatter(train_df['GrLivArea'], train_df['SalePrice'])
# Scatter plot between Basement area and Sales price

plt.scatter(train_df[train_df['TotalBsmtSF']>0]['TotalBsmtSF'], train_df[train_df['TotalBsmtSF']>0]['SalePrice'])
train_df = pd.get_dummies(train_df)
train_df.dtypes