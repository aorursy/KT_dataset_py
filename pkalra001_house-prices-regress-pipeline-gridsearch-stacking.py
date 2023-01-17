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
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

train_df.head()
train_id = train_df['Id']

test_id = test_df['Id']

train_df.drop('Id', axis = 1, inplace = True)

test_df.drop('Id', axis = 1, inplace = True)

train_df.info()
#Read from a public notebook that column GrLivArea have ouliers as it was mentioned in the original data documentation which isnt available anymore

fig, ax = plt.subplots()

ax.scatter(x = train_df['GrLivArea'], y = train_df['SalePrice'])

plt.show()
#We can see the two big outliers..lets delete them

train_df.drop(train_df[(train_df['SalePrice'] < 200000) & (train_df['GrLivArea'] > 4500)].index, inplace = True)

fig, ax = plt.subplots()

ax.scatter(x = train_df['GrLivArea'], y = train_df['SalePrice'])

plt.show()
#Combining data

all_data = pd.concat((train_df, test_df)).reset_index(drop = True)

all_data.drop('SalePrice', axis = 1, inplace = True)

print(all_data.shape)
all_miss_data = ((all_data.isnull().sum())/len(all_data))* 100

all_miss_data = all_miss_data.drop(all_miss_data[all_miss_data == 0].index).sort_values(ascending = False)

print(all_miss_data)
plt.subplots(figsize = (12, 9))

sns.heatmap(train_df.corr(), square = True)

#nothing unexpected can be seen here
#lets fill the houses with no pool with none value

all_data['PoolQC'] = all_data['PoolQC'].fillna('None')

#house with no misc feauture

all_data['MiscFeature'] = all_data['MiscFeature'].fillna('None')

#house with no alley

all_data['Alley'] = all_data['Alley'].fillna('None')

#houses with no fence

all_data['Fence'] = all_data['Fence'].fillna('None')

#houses with no fire place

all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna('None')
'''For LotFrontage feature we assume that the houses in the same neighbourhood have similar frontage so for every missing 

value we replace it with the median frontage value of the neighbourhood'''

all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x : x.fillna(x.median()))



# Filling the four 'garage' categorical variables with None

for col in ('GarageFinish', 'GarageQual', 'GarageCond', 'GarageType'):

    all_data[col] = all_data[col].fillna('None')

# Filling the three 'garage' continous variables with 0

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    all_data[col] = all_data[col].fillna(0)
#For basement categorical features

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    all_data[col] = all_data[col].fillna('None')

#For basement continous features

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    all_data[col] = all_data[col].fillna(0)
#If the house has no masonary veneer

all_data['MasVnrType'] = all_data['MasVnrType'].fillna('None')

all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)
print(all_data['MSZoning'].value_counts())

all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
print(all_data['Utilities'].value_counts())

all_data = all_data.drop(['Utilities'], axis=1)
#ames data description says NA in the functional column means typical

print(all_data['Functional'].value_counts())

all_data['Functional'] = all_data["Functional"].fillna("Typ")
print(all_data['Electrical'].value_counts())

all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
print(all_data['KitchenQual'].value_counts())

all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
#replacing with mode values

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
#na in subclass means no class

print(all_data['MSSubClass'].value_counts())

all_data['MSSubClass'] = all_data['MSSubClass'].fillna('None')
all_miss_data = ((all_data.isnull().sum())/len(all_data))* 100

all_miss_data = all_miss_data.drop(all_miss_data[all_miss_data == 0].index).sort_values(ascending = False)

print(all_miss_data)
print(all_data.info())
all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)

#print(all_data['OverallQual'].value_counts())

all_data['OverallQual'] = all_data['OverallQual'].astype(str)

#print(all_data['OverallCond'].value_counts())

all_data['OverallCond'] = all_data['OverallCond'].astype(str)

all_data['YearBuilt'] = all_data['YearBuilt'].astype(str)

#print(all_data['YearRemodAdd'].value_counts())

all_data['YearRemodAdd'] = all_data['YearRemodAdd'].astype(str)
from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')

# process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(all_data[c].values)) 

    all_data[c] = lbl.transform(list(all_data[c].values))