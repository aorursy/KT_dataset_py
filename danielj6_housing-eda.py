# Imports for data wrangling and visualization

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from scipy.stats import norm
# Set pandas options to not truncate the display of large dataframes (so that we can see info for all features)

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
# Don't display warning when many figues are open

plt.rcParams.update({'figure.max_open_warning': 0})
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
print(train.shape)

print(test.shape)
summary = pd.DataFrame()

summary['dtype'] = train.dtypes

summary['unique'] = train.nunique(axis=0)

summary['missing'] = train.isnull().sum()

summary['mean'] = train.mean()

summary['std'] = train.std()

summary['mode'] = train.mode(axis=0).iloc[0]

summary['skew'] = train.skew()

summary = summary.sort_values('dtype')

summary['number'] = np.arange(len(summary))+1

summary
print('Cardinality of categorical features:')

print(train.select_dtypes(exclude=np.number).nunique().sort_values(ascending=False).to_string())
train.describe()
train.head(5)
# make lists of categorical and numeric features

categorical = [i for i in train.columns if train.dtypes[i] == 'object']

numerical = [f for f in train.columns if train.dtypes[f] != 'object']

numerical.remove('SalePrice')

#numerical.remove('Id')
# Most common values of categorical features

for i in train.select_dtypes(exclude=np.number).columns:

    print(train[i].value_counts()[:5])
# Documentation indicates that there are some outliers that can be seen in plot of GrLivArea vs SalePrice

#plt.scatter(train['GrLivArea'], train['SalePrice'])

sns.scatterplot(x='GrLivArea', y='SalePrice', data=train)
missing_percent = (train.isnull().sum()/len(train)) * 100

plt.figure(figsize=(16, 3))

missing_percent = missing_percent[missing_percent >  0]

missing_percent.sort_values(ascending=False).plot.bar(rot=90)
sns.distplot(train['SalePrice'], fit=norm)
sns.distplot(np.log1p(train['SalePrice']), fit=norm)
for feature in numerical:

    f = plt.figure()

    sns.distplot(train[feature].astype(float), bins=100, kde=False)
for feature in numerical:

    f, axs = plt.subplots(1, 2, sharey=True)

    isnotnan = ~np.isnan(train[feature])

    axs[0].hist2d(train[feature][isnotnan], train['SalePrice'][isnotnan], bins=50, cmap='Greys')

    axs[0].set_xlabel(feature)

    axs[0].set_ylabel('SalePrice')

    sns.regplot(x=feature, y='SalePrice', data=train, x_bins=20, ax=axs[1])
for var in categorical:

    f, axs = plt.subplots(1, 3, figsize=(12, 5))

    order = train.groupby(by=[var])['SalePrice'].median().sort_values(ascending=True).index

    sns.countplot(x=var, data=train, order=order, ax=axs[0])

    sns.pointplot(x=var, y='SalePrice', data=train, order=order, ax=axs[1])

    sns.boxplot(x=var, y='SalePrice', data=train, order=order, ax=axs[2])

    axs[1].set_ylim((0, 800000))

    axs[2].set_ylim((0, 800000))

    plt.tight_layout()

    for ax in axs:

        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
f = plt.figure(figsize=(5, 8))

train.corr(method='pearson')['SalePrice'].sort_values(ascending=True).drop(['SalePrice'], axis=0).plot.barh()

plt.title('Pearson Correlation of Features with Target')

f = plt.figure(figsize=(5, 8))

train.corr(method='spearman')['SalePrice'].sort_values(ascending=True).drop(['SalePrice'], axis=0).plot.barh()

plt.title('Spearman Correlation of Features with Target')
plt.figure(figsize=(10, 8))

sns.heatmap(train.drop(['SalePrice'], axis=1).corr(), annot=False, cmap='coolwarm', fmt='.2f', center=0)

plt.tight_layout()

plt.title('Pearson Correlation of Features')