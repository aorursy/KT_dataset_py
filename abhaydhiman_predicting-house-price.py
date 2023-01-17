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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
train_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



df = [train_df, test_df]
train_df.shape
train_df.head()
total_null = dict(train_df.isnull().sum())

for i, j in total_null.items():

    print(i, ' ---> ', j)
total_null = dict(test_df.isnull().sum())

for i, j in total_null.items():

    print(i, ' ---> ', j)
sns.heatmap(train_df.isnull(), yticklabels=False, cbar=False)
for dataset in df:

    dataset.drop(['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)
train_df.info()
for dataset in df:

    dataset['LotFrontage'] = dataset['LotFrontage'].fillna(dataset['LotFrontage'].mean())#float

    dataset['BsmtQual'] = dataset['BsmtQual'].fillna(dataset['BsmtQual'].mode()[0])

    dataset['BsmtCond'] = dataset['BsmtCond'].fillna(dataset['BsmtCond'].mode()[0])

    dataset['BsmtExposure'] = dataset['BsmtExposure'].fillna(dataset['BsmtExposure'].mode()[0])

    dataset['BsmtFinType1'] = dataset['BsmtFinType1'].fillna(dataset['BsmtFinType1'].mode()[0])

    dataset['MasVnrType'] = dataset['MasVnrType'].fillna(dataset['MasVnrType'].mode()[0])

    dataset['MasVnrArea'] = dataset['MasVnrArea'].fillna(dataset['MasVnrArea'].mean())#float

    dataset['BsmtFinType2'] = dataset['BsmtFinType2'].fillna(dataset['BsmtFinType2'].mode()[0])

    dataset['Electrical'] = dataset['Electrical'].fillna(dataset['Electrical'].mode()[0])

    dataset['FireplaceQu'] = dataset['FireplaceQu'].fillna(dataset['FireplaceQu'].mode()[0])

    dataset['GarageType'] = dataset['GarageType'].fillna(dataset['GarageType'].mode()[0])

    dataset['GarageYrBlt'] = dataset['GarageYrBlt'].fillna(dataset['GarageYrBlt'].mean())#float

    dataset['GarageFinish'] = dataset['GarageFinish'].fillna(dataset['GarageFinish'].mode()[0])

    dataset['GarageQual'] = dataset['GarageQual'].fillna(dataset['GarageQual'].mode()[0])

    dataset['GarageCond'] = dataset['GarageCond'].fillna(dataset['GarageCond'].mode()[0])

    #In test case

    dataset['GarageCars'] = dataset['GarageCars'].fillna(dataset['GarageCars'].mean())

    dataset['GarageArea'] = dataset['GarageArea'].fillna(dataset['GarageArea'].mean())

    dataset['SaleType'] = dataset['SaleType'].fillna(dataset['SaleType'].mode()[0])

    dataset['Functional'] = dataset['Functional'].fillna(dataset['Functional'].mode()[0])

    dataset['KitchenQual'] = dataset['KitchenQual'].fillna(dataset['KitchenQual'].mode()[0])

    dataset['BsmtHalfBath'] = dataset['BsmtHalfBath'].fillna(dataset['BsmtHalfBath'].mean())

    dataset['BsmtFullBath'] = dataset['BsmtFullBath'].fillna(dataset['BsmtFullBath'].mean())

    dataset['TotalBsmtSF'] = dataset['TotalBsmtSF'].fillna(dataset['TotalBsmtSF'].mean())

    dataset['BsmtUnfSF'] = dataset['BsmtUnfSF'].fillna(dataset['BsmtUnfSF'].mean())

    dataset['BsmtFinSF2'] = dataset['BsmtFinSF2'].fillna(dataset['BsmtFinSF2'].mean())

    dataset['BsmtFinSF1'] = dataset['BsmtFinSF1'].fillna(dataset['BsmtFinSF1'].mean())

    dataset['Exterior2nd'] = dataset['Exterior2nd'].fillna(dataset['Exterior2nd'].mode()[0])

    dataset['Exterior1st'] = dataset['Exterior1st'].fillna(dataset['Exterior1st'].mode()[0])

    dataset['Utilities'] = dataset['Utilities'].fillna(dataset['Utilities'].mode()[0])

    dataset['MSZoning'] = dataset['MSZoning'].fillna(dataset['MSZoning'].mode()[0])
dataframe = pd.concat([train_df, test_df])
dataframe.shape
dataframe2 = pd.get_dummies(dataframe, drop_first=True)
dataframe2.shape
train_df = dataframe2.iloc[: 1460, :]

test_df = dataframe2.iloc[1460: , :]
train_df.shape
test_df.shape
X = train_df.drop('SalePrice', axis=1)

y = train_df['SalePrice']

test_df = test_df.drop('SalePrice', axis=1)
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV
tree_regressor = DecisionTreeRegressor()

tree_regressor.fit(X, y)

tree_regressor.score(X, y)
forest_regressor = RandomForestRegressor()

forest_regressor.fit(X, y)

forest_regressor.score(X, y)
Xgb_regressor = XGBRegressor(n_jobs=-1)

Xgb_regressor.fit(X, y)
param_test = {

    'max_depth': [3, 5, 7, 9],

    'min_child_weight': [1, 3, 5],

#     'gamma': [i/10.0 for i in range(0,5)],

#     'subsample': [i/10.0 for i in range(6,10)],

#     'colsample_bytree': [i/10.0 for i in range(6,10)],

#     'reg_alpha': [0, 0.001, 0.005, 0.01, 1, 100],

    'learning_rate': [0.1, 0.2, 0.3],

    'n_estimators': [100, 400, 600, 900, 1100]

}
gsearch = GridSearchCV(estimator = Xgb_regressor, 

                       param_grid = param_test,

                       scoring='neg_mean_squared_log_error',

                       n_jobs=-1,

                       cv=5)
# gsearch.fit(X, y)
# gsearch.best_estimator_
Xgb_regressor = XGBRegressor(base_score=0.25, colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,

             importance_type='gain', learning_rate=0.1, max_delta_step=0, max_depth=3,

             min_child_weight=1, n_estimators=600, n_jobs=-1, num_parallel_tree=1,

             objective='reg:squarederror', random_state=0, reg_alpha=0,

             reg_lambda=1, scale_pos_weight=1, subsample=1)





Xgb_regressor.fit(X, y)
y_pred = Xgb_regressor.predict(test_df)
sample_sub = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
datasets = pd.DataFrame({

    'Id': sample_sub['Id'],

    'SalePrice': y_pred

})
datasets.to_csv('gender_submission3.csv', index=False)