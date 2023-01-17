import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 2000)

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
houseRaw = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
houseSub = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
house = houseRaw.copy()
house.columns
house.head(3)
totalMissing = house.isnull().sum().sort_values(ascending = False)
percMissing = house.isnull().sum() / house.isnull().count().sort_values(ascending = False)
missing = pd.concat([totalMissing, percMissing], axis = 1, keys = ['total #', '%'])
missing[missing['total #'] > 0]
# LotFrontage
print('The correlation matrix between {c} and SalePrice is {n}'.format(c = 'LotFrontage', n = house[['LotFrontage', 'SalePrice']].corr()))
plt.figure(figsize = (10, 8))
sns.scatterplot(x = 'SalePrice', y = 'LotFrontage', data = house)
# MasVnrArea
print('The correlation matrix between {c} and SalePrice is {n}'.format(c = 'MasVnrArea', n = house[['MasVnrArea', 'SalePrice']].corr()))
plt.figure(figsize = (10, 8))
sns.scatterplot(x = 'SalePrice', y = 'MasVnrArea', data = house)
# final data set
house = house.drop(columns = 'GarageYrBlt')
house = house.drop(house.loc[house['Electrical'].isnull()].index)
house['LotFrontage'].fillna(house['LotFrontage'].median(), inplace = True)
house['MasVnrArea'].fillna(house['MasVnrArea'].median(), inplace = True)
colums_to_fill = missing.drop(['GarageYrBlt', 'Electrical'], axis = 0)[missing['total #'] > 0].index
for i in range(len(colums_to_fill)):
    house.loc[house[colums_to_fill[i]].isnull(), colums_to_fill[i]] = 'No Feature'
house['LotFrontage'] = house['LotFrontage'].astype('float')
house['MasVnrArea'] = house['MasVnrArea'].astype('float')
house.isnull().sum().sum()
# Look through the data attributes and dictionary and classify data into categorical and numeric types
IdCol = ['Id']
label = ['SalePrice']
num = [
    'LotArea', 'LotFrontage', 'MasVnrArea', 'YearBuilt', 'YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
    'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
    'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'OverallQual', 'OverallCond'
]
cat = house.drop(columns = num + IdCol + label, axis = 1).columns
house[cat] = house[cat].astype('category')
corr_matrix = house.drop(columns = ['Id'])[num + label].corr()
plt.figure(figsize = (15, 12))
sns.heatmap(corr_matrix, vmin = -1, vmax = 1, square = True, cmap = sns.diverging_palette(225, 10, n = 7))
house['DaySold'] = 1
ddtt = house[['YrSold', 'MoSold', 'DaySold']].rename(columns = {'YrSold': 'year', 'MoSold': 'month', 'DaySold': 'day'})
house['year_month'] = pd.to_datetime(ddtt)
plt.figure(figsize = (30, 5))
sns.lineplot(x = 'year_month', y = 'SalePrice', data = house)
plt.figure(figsize = (30, 5))
sns.lineplot(x = 'YearBuilt', y = 'SalePrice', data = house)
corr_matrix_top = house[['OverallQual', 'MasVnrArea', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'SalePrice']].corr()
plt.figure(figsize = (15, 12))
sns.heatmap(corr_matrix_top, vmin = -1, vmax = 1, square = True, annot = True, cmap = sns.diverging_palette(225, 10, n = 7))
sns.set()
sns.pairplot(house[['OverallQual', 'MasVnrArea', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', 'GrLivArea', 'FullBath', 'Fireplaces', 'GarageCars', 'GarageArea', 'SalePrice']], height = 2.5)
plt.show()
cat
f, axes = plt.subplots(1, 2, figsize = (30, 10))
sns.set_style('whitegrid')
sns.boxplot(x = 'MSSubClass', y = 'SalePrice', data = house, ax = axes[0])
sns.boxplot(x = 'MSZoning', y = 'SalePrice', data = house, ax = axes[1])
plt.figure(figsize = (20, 10))
sns.boxplot(x = 'Neighborhood', y = 'SalePrice', data = house)
plt.xticks(rotation = 45)
f, axes = plt.subplots(1, 2, figsize = (30, 10))
sns.boxplot(x = 'Condition1', y = 'SalePrice', data = house, ax = axes[0])
sns.boxplot(x = 'Condition2', y = 'SalePrice', data = house, ax = axes[1])
f, axes = plt.subplots(2, 2, figsize = (30, 20))
sns.boxplot(x = 'SaleCondition', y = 'SalePrice', data = house, ax = axes[0, 0])
sns.boxplot(x = 'ExterQual', y = 'SalePrice', data = house, ax = axes[0, 1])
sns.boxplot(x = 'GarageQual', y = 'SalePrice', data = house, ax = axes[1, 0])
sns.boxplot(x = 'BsmtQual', y = 'SalePrice', data = house, ax = axes[1, 1])