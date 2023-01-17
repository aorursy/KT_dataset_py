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
df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df.info()
df.head(2)
df.describe()
df.isnull().sum()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
plt.figure(figsize = (20,6))

sns.heatmap(df.isnull(), cbar = False)
df.columns[df.isnull().mean() > 0.2]
df.drop(['Id','Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis = 1, inplace = True)
df.head(2)
plt.figure(figsize = (20,6))

sns.heatmap(df.isnull(), cbar = False);
df.LotFrontage.isnull().sum()/len(df.LotFrontage)
corr = df.corr()
plt.figure(figsize=(30,15))

sns.heatmap(corr, square=True, vmin = -1, vmax = 1,cmap = 'coolwarm', linewidths=.5);
df.LotFrontage.fillna(df.LotFrontage.mean(), inplace = True)
plt.figure(figsize = (20,6))

sns.heatmap(df.isnull(), cbar = False);
df.columns[df.isnull().sum() > 0]
df[['MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure',

       'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'GarageType',

       'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond']].isnull().sum()/len(df[['MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure',

       'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'GarageType',

       'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond']])
df.dropna(inplace = True)
df
df.select_dtypes('object').columns
df1 = pd.get_dummies(df, columns=['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',

       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',

       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',

       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',

       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',

       'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',

       'PavedDrive', 'SaleType', 'SaleCondition'], drop_first=True)
print(df.columns.values)
print(df1.columns.values)
corr_matrix = df1.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.70)]
to_drop.remove('SalePrice')
to_drop
df1.head(2)
train = df1.drop(to_drop,axis=1)
train.columns.values
from sklearn.model_selection import train_test_split

X = train.drop('SalePrice', axis = 1)

y = train.SalePrice
from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor()
RF.fit(X,y)
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
test.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis = 1, inplace = True)
test.LotFrontage.fillna(test.LotFrontage.mean(), inplace = True)
test.columns[test.isnull().sum() > 0]
test.BsmtFullBath.dtype
test.GarageFinish.isnull().sum()
test.SaleType.value_counts()
test.MSZoning.fillna('RL', inplace = True)

test.Utilities.fillna('AllPub', inplace = True)

test.Exterior1st.fillna('VinylSd', inplace = True)

test.Exterior2nd.fillna('VinylSd', inplace = True)

test.MasVnrType.fillna('None', inplace = True)

test.MasVnrArea.fillna(test.MasVnrArea.mean(), inplace = True)

test.BsmtQual.fillna('TA', inplace = True)

test.BsmtCond.fillna('TA', inplace = True)

test.BsmtExposure.fillna('No', inplace = True)

test.BsmtFinType1.fillna('GLQ', inplace = True)

test.BsmtFinSF1.fillna(test.BsmtFinSF1.mean(), inplace = True)

test.BsmtFinType2.fillna('Unf', inplace = True) 

test.BsmtFinSF2.fillna(test.BsmtFinSF2.mean(), inplace = True)

test.BsmtUnfSF.fillna(test.BsmtUnfSF.mean(), inplace = True)

test.TotalBsmtSF.fillna(test.TotalBsmtSF.mean(), inplace = True)

test.BsmtFullBath.fillna(0.0, inplace = True)

test.BsmtHalfBath.fillna(0.0, inplace = True)

test.KitchenQual.fillna('TA', inplace = True) 

test.Functional.fillna('Typ', inplace = True)

test.GarageType.fillna('Attchd', inplace = True)

test.GarageYrBlt.fillna(2005, inplace = True)

test.GarageFinish.fillna('Unf', inplace = True) 

test.GarageCars.fillna(2, inplace = True)

test.GarageArea.fillna(test.GarageArea.mean(), inplace = True)

test.GarageQual.fillna('TA', inplace = True)

test.GarageCond.fillna('TA', inplace = True)

test.SaleType.fillna('WD', inplace = True)
test.info()
test.head(2)
test1 = pd.get_dummies(test, columns=['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',

       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',

       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',

       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',

       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',

       'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',

       'PavedDrive', 'SaleType', 'SaleCondition'], drop_first=True)
test1.head(2)
to_drop
missing_cols = set( train.columns ) - set( test.columns )
extra_cols = set( test1.columns ) - set( train.columns )
extra_cols
test2 = test1.drop(extra_cols, axis = 1)
test2
missing_cols = set( train.columns ) - set( test2.columns )
missing_cols
for c in missing_cols:

    test2[c] = 0
test_final = test2[train.columns]
test_final.columns.values
train.columns.values
test_final = test_final.drop('SalePrice', axis =1)
predictions = RF.predict(test_final)
outputRF = pd.DataFrame({'Id': test.Id, 'SalePrice': predictions})
outputRF.to_csv('house_submission2.csv', index=False) 