# Required library imports

import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt
# Reading the training data

data = pd.read_csv('../input/train.csv')

data.head()
data.isnull().sum()
_min = data['SalePrice'].min()

print("Min Selling Price is :",_min)

_avg = data['SalePrice'].mean()

print("Average Selling Price is :",_avg)

_median = data['SalePrice'].median()

print("Median Selling Price is :",_median)

_max = data['SalePrice'].max()

print("Max Selling Price is :",_max)
data['SalePrice'].plot.hist(bins=30,edgecolor='black',color='green')

fig=plt.gcf()

fig.set_size_inches(25,15)

x_range = range(0,750000,25000)

plt.xticks(x_range)

plt.show()
# Convert target variable to categorical like low,medium,high price and analyse

data['SalePrice_Cat'] = 0

data.loc[data.SalePrice <= 100000,'SalePrice_Cat'] = 0

data.loc[(data.SalePrice > 100000) & (data.SalePrice <=200000),'SalePrice_Cat'] = 1

data.loc[data.SalePrice > 300000,'SalePrice_Cat'] = 2

data.head(2)
# Binning based on domain

data['SalePrice_Cat'].value_counts()
sns.countplot('SalePrice_Cat',data=data,color='blue')

plt.show()
data['SalePrice_qcut_Cat'] = 0

data['SalePrice_qcut_Cat'] = pd.qcut(data['SalePrice'],20)

data.head(2)
# Binning based on quantiles

data['SalePrice_qcut_Cat'].value_counts()
fig, axes = plt.subplots(1,2,figsize=(18,5))

sns.countplot('MSSubClass',data=data,ax=axes[0],color='blue')

sns.countplot('MSSubClass',hue='SalePrice_Cat',data=data,ax=axes[1])

plt.show()
pd.crosstab([data.SalePrice_Cat],[data.MSSubClass],margins=True).style.background_gradient(cmap='summer_r')
pd.crosstab([data.SalePrice_qcut_Cat],[data.MSSubClass],margins=True).style.background_gradient('summer_r')
# Checking for any null values

print(data['GrLivArea'].isnull().sum())

print(data['OverallQual'].isnull().sum())

print(data['OverallCond'].isnull().sum())

print(data['Neighborhood'].isnull().sum())

# There aren't any N.A values in above columns
# OverallQual

print("OverallQual")

print(data['OverallQual'].value_counts())
# OverallCond

print("OverallCond")

print(data['OverallCond'].value_counts())
# Neighborhood

print("Neighborhood")

print(data['Neighborhood'].value_counts())
pd.crosstab([data.SalePrice_Cat],[data.OverallQual],margins=True).style.background_gradient('summer_r')
fig, axes = plt.subplots(1,2,figsize=(18,5))

sns.countplot('OverallQual',data=data,ax=axes[0])

sns.countplot('OverallQual',hue='SalePrice_Cat',data=data,ax=axes[1])

plt.show()
pd.crosstab([data.SalePrice_Cat],[data.OverallCond],margins=True).style.background_gradient('summer_r')
fig, axes = plt.subplots(1,2,figsize=(18,5))

sns.countplot('OverallCond',data=data,ax=axes[0])

sns.countplot('OverallCond',hue='SalePrice_Cat',data=data,ax=axes[1])

plt.show()
pd.crosstab([data.SalePrice_Cat],[data.Neighborhood],margins=True).style.background_gradient('summer_r')
fig, axes = plt.subplots(1,2,figsize=(18,5))

sns.countplot('Neighborhood',data=data,ax=axes[0])

sns.countplot('Neighborhood',hue='SalePrice_Cat',data=data,ax=axes[1])

plt.show()
_min_grliv_area = data['GrLivArea'].min()

print("Min GrLiveArea is :",_min_grliv_area)

_avg_grliv_area = data['GrLivArea'].mean()

print("Average GrLiveArea is :",_avg_grliv_area)

_median_grliv_area = data['GrLivArea'].median()

print("Median GrLiveArea is :",_median_grliv_area)

_max_grliv_area = data['GrLivArea'].max()

print("Max GrLiveArea is :",_max_grliv_area)
data['GrLivArea'].plot.hist(bins=20,edgecolor='black',color='yellow')

plt.xticks(range(0,5642,275))

plt.show()
data['GrLivArea_Cat'] = 0

data['GrLivArea_Cat'] = pd.qcut(data['GrLivArea'],10)

data.head()
data['GrLivArea_Cat'].value_counts()
pd.crosstab([data.GrLivArea_Cat],[data.SalePrice_Cat],margins=True).style.background_gradient('summer_r')
data['YearRemodAdd'].isnull().any()

# There isn't any N.A value in the column YearRemodAdd
print(data['YearRemodAdd'].min())

print(data['YearRemodAdd'].max())
data['RemodelYear_Cat'] = 0

data.loc[data.YearRemodAdd <= 1950, 'RemodelYear_Cat'] = 0

data.loc[(data.YearRemodAdd > 1950) & (data.YearRemodAdd <= 1960), 'RemodelYear_Cat'] = 1

data.loc[(data.YearRemodAdd > 1960) & (data.YearRemodAdd <= 1970), 'RemodelYear_Cat'] = 2

data.loc[(data.YearRemodAdd > 1970) & (data.YearRemodAdd <= 1980), 'RemodelYear_Cat'] = 3

data.loc[(data.YearRemodAdd > 1980) & (data.YearRemodAdd <= 1990), 'RemodelYear_Cat'] = 4

data.loc[(data.YearRemodAdd > 1990) & (data.YearRemodAdd <= 2000), 'RemodelYear_Cat'] = 5

data.loc[(data.YearRemodAdd > 2000) & (data.YearRemodAdd <= 2010), 'RemodelYear_Cat'] = 6

data.head()
data['RemodelYear_Cat'].value_counts()
fig, axes = plt.subplots(1,2,figsize=(18,5))

sns.countplot('RemodelYear_Cat',data=data,ax=axes[0])

sns.countplot('RemodelYear_Cat',hue='SalePrice_Cat',data=data,ax=axes[1])

plt.show()
data['RoofStyle'].isnull().any()

# No N.A value
data['RoofStyle'].value_counts()
data['RoofStyle'].value_counts().plot.pie()

plt.show()
fig, axes = plt.subplots(1,2,figsize=(18,5))

sns.countplot('RoofStyle',data=data,ax=axes[0])

sns.countplot('RoofStyle',hue='SalePrice_Cat',data=data,ax=axes[1])

plt.show()
data['SaleCondition'].isnull().any()

# Again no N.A values
data['SaleCondition'].value_counts()
fig, axes = plt.subplots(1,2,figsize=(18,5))

sns.countplot('SaleCondition',data=data,ax=axes[0])

sns.countplot('SaleCondition',hue='SalePrice_Cat',data=data,ax=axes[1])

plt.show()
data['YrSold'].isnull().any()

# Again no N.A values
data['YrSold'].value_counts()
fig, axes = plt.subplots(1,2,figsize=(18,5))

sns.countplot('YrSold',data=data,ax=axes[0])

sns.countplot('YrSold',hue='SalePrice_Cat',data=data,ax=axes[1])

plt.show()
data['LotShape'].value_counts()
fig, axes = plt.subplots(1,2,figsize=(18,5))

sns.countplot('LotShape',data=data,ax=axes[0])

sns.countplot('LotShape',hue='SalePrice_Cat',data=data,ax=axes[1])

plt.show()
data['YearBuilt'].isnull().any()

# No N.A. values, Great!
print(data['YearBuilt'].min())

print(data['YearBuilt'].max())

# data['YearBuilt'].value_counts()
data['YearBuilt_Cat'] = 0

data.loc[data.YearBuilt <= 1880,'YearBuilt_Cat'] = 0

data.loc[(data.YearBuilt > 1880) & (data.YearBuilt <= 1890),'YearBuilt_Cat'] = 1

data.loc[(data.YearBuilt > 1890) & (data.YearBuilt <= 2000),'YearBuilt_Cat'] = 2

data.loc[(data.YearBuilt > 2000) & (data.YearBuilt <= 2010),'YearBuilt_Cat'] = 3

data.loc[data.YearBuilt > 2010,'YearBuilt_Cat'] = 4

data.head(2)
fig, axes = plt.subplots(1,2,figsize=(18,5))

sns.countplot('YearBuilt_Cat',data=data,ax=axes[0],color='blue')

sns.countplot('YearBuilt_Cat',hue='SalePrice_Cat',data=data,ax=axes[1])

plt.show()