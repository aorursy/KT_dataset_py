import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import seaborn as sns

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
test_path='../input/house-prices-advanced-regression-techniques/test.csv'

train_path = '../input/house-prices-advanced-regression-techniques/train.csv'



test = pd.read_csv(test_path, index_col= 'Id')

train = pd.read_csv(train_path, index_col = 'Id')
print("test shape:",test.shape)

print("train shape:",train.shape)
train.head()
missing_num = train.isnull().sum()

missing_cols = missing_num[missing_num>0]

print(missing_cols)

print("--- as percentage ---")

print(missing_cols/1460*100)
sns.distplot(train['SalePrice'])

plt.show()
train['SalePrice'].describe()
plt.figure(figsize=(15,15))

corrmat = train.corr()

sns.heatmap(corrmat, vmax=.8, square=True)
plt.figure(figsize=(20,20))

k = 20 # take the top k most correlated with SalePrice

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': k}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
# potentially useful columns: ['LotFrontage', 'OverallQual', 'YearBuilt', 'MasVnrArea', 'TotalBsmtSF', 'GrLivArea', 'FullBath', 'HalfBath', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF','SalePrice']

explore_cols=['MasVnrArea', 'TotalBsmtSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF']

explore_corr = train[explore_cols].corr()

sns.heatmap(explore_corr, vmax=.8, square=True)

plt.show()
train['RecSF'] = train['WoodDeckSF'] + train['OpenPorchSF']

train['HouseSF'] = train['TotalBsmtSF'] + train['GrLivArea']

train['Baths'] = train['FullBath'] + (train['HalfBath']/2)
useful_num_cols = ['OverallQual', 'YearBuilt', 'MasVnrArea', 'RecSF','HouseSF','Baths', 'GarageArea','SalePrice']

sns.pairplot(train[useful_num_cols])

plt.show();
train[train['LotFrontage']>=300]

# 935 and 1299
sns.violinplot(x=train['SaleCondition'], y= train["SalePrice"])

plt.show()

# Looks like partial sell has two population?
# These two points are because of partial sale but still, very off

train[train['HouseSF']>=5200]

# 524, 1299
sns.violinplot(x=train['LandContour'], y= train["SalePrice"])

plt.show()
train.drop([524,935,1299], inplace=True)
# useful_num_cols = ['OverallQual', 'YearBuilt', 'MasVnrArea', 'RecSF','HouseSF','Baths', 'GarageArea','SalePrice']



cat_cols = train.select_dtypes(include=['object']).columns

# print(cat_cols)# 43 total



fig, ax = plt.subplots(11,4, figsize=(15,40), dpi=100)

i=1

for col in cat_cols:

    plt.subplot(11,4,i)

    sns.violinplot(x=train[col], y= train["SalePrice"])

    i+=1

    

plt.show()

    

pontential_cat_col=['MSZoning','Neighborhood','Condition1', 'HouseStyle','Exterior1st','MasVnrType','SaleType','SaleCondition']

# 'ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu' - make it into 1 to 5, numerical

"""

       Ex	Excellent - 5

       Gd	Good - 4

       TA	Typical/Average - 3

       Fa	Fair - 2

       Po	Poor - 1

"""

# make 'CentralAir' and 'Pool' binary

# # Heating - get rid of Floor furnace?

# 'MiscFeature' - might be adding bonus money if Gar2 or Shed present

# FireplaceQual -- only matters if it is Excellet