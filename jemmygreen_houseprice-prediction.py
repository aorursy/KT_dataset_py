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
train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv',index_col=0)

test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv', index_col=0)

train_df.head()
missing_data = train_df.isnull().sum().sort_values(ascending=False).head(20)

percentage_missing_data = train_df.isnull().sum()/train_df.isnull().count().sort_values(ascending = False) * 100

missing_values = pd.concat([missing_data, percentage_missing_data], axis=1, keys=['missing_data', 'percentage'])

missing_values.head(20)
train_df.drop(['MiscFeature', 'PoolQC', 'Alley', 'Fence'], axis=1, inplace=True)

train_df.shape
# Filling the numerical Null values with mean

train_df['LotFrontage'] = train_df['LotFrontage'].fillna(train_df['LotFrontage'].mean())
## Filling the categorical null values with mode

train_df['GarageCond'] = train_df['GarageCond'].fillna(train_df['GarageCond'].mode()[0])

train_df['FireplaceQu'] = train_df['FireplaceQu'].fillna(train_df['FireplaceQu'].mode()[0])

train_df['GarageType'] = train_df['GarageType'].fillna(train_df['GarageType'].mode()[0])

train_df['GarageYrBlt'] = train_df['GarageYrBlt'].fillna(train_df['GarageYrBlt'].mode()[0])

train_df['GarageFinish'] = train_df['GarageFinish'].fillna(train_df['GarageFinish'].mode()[0])

train_df['GarageQual'] = train_df['GarageQual'].fillna(train_df['GarageQual'].mode()[0])

train_df['BsmtExposure'] = train_df['BsmtExposure'].fillna(train_df['BsmtExposure'].mode()[0])

train_df['BsmtFinType2'] = train_df['BsmtFinType2'].fillna(train_df['BsmtFinType2'].mode()[0])

train_df['BsmtFinType1'] = train_df['BsmtFinType1'].fillna(train_df['BsmtFinType1'].mode()[0])

train_df['BsmtCond'] = train_df['BsmtCond'].fillna(train_df['BsmtCond'].mode()[0])

train_df['BsmtQual'] = train_df['BsmtQual'].fillna(train_df['BsmtQual'].mode()[0])

train_df['MasVnrArea'] = train_df['MasVnrArea'].fillna(train_df['MasVnrArea'].mode()[0])

train_df['MasVnrType'] = train_df['MasVnrType'].fillna(train_df['MasVnrType'].mode()[0])

train_df['Electrical'] = train_df['Electrical'].fillna(train_df['Electrical'].mode()[0])
# Checking the shape of the train dataset

train_df.shape
## Handling the test dataset

test_df.head()
missing_test_data = test_df.isnull().sum().sort_values(ascending=False)

percentage_mtd = test_df.isnull().sum()/test_df.isnull().count().sort_values(ascending=False)*100

missing_test_values = pd.concat([missing_test_data, percentage_mtd], axis=1, keys=['missing_test_data', 'percentage'])

missing_test_values.head(34)
test_df.info()


test_df.drop(['PoolQC', 'Fence', 'Alley', 'MiscFeature'], axis=1, inplace=True)
test_df.shape
## Fill Missing Numerical Values with mean

test_df['LotFrontage'] = test_df['LotFrontage'].fillna(test_df['LotFrontage'].mean())
## Fill Missing Categorical Values with mode

test_df['FireplaceQu'] = test_df['FireplaceQu'].fillna(test_df['FireplaceQu'].mode()[0])

test_df['GarageCond'] = test_df['GarageCond'].fillna(test_df['GarageCond'].mode()[0])

test_df['GarageFinish'] = test_df['GarageFinish'].fillna(test_df['GarageFinish'].mode()[0])

test_df['GarageYrBlt'] = test_df['GarageYrBlt'].fillna(test_df['GarageYrBlt'].mode()[0])

test_df['GarageQual'] = test_df['GarageQual'].fillna(test_df['GarageQual'].mode()[0])

test_df['GarageType'] = test_df['GarageType'].fillna(test_df['GarageType'].mode()[0])

test_df['BsmtCond'] = test_df['BsmtCond'].fillna(test_df['BsmtCond'].mode()[0])

test_df['BsmtExposure'] = test_df['BsmtExposure'].fillna(test_df['BsmtExposure'].mode()[0])

test_df['BsmtQual'] = test_df['BsmtQual'].fillna(test_df['BsmtQual'].mode()[0])

test_df['BsmtFinType1'] = test_df['BsmtFinType1'].fillna(test_df['BsmtFinType1'].mode()[0])

test_df['BsmtFinType2'] = test_df['BsmtFinType2'].fillna(test_df['BsmtFinType2'].mode()[0])

test_df['MasVnrType'] = test_df['MasVnrType'].fillna(test_df['MasVnrType'].mode()[0])

test_df['MasVnrArea'] = test_df['MasVnrArea'].fillna(test_df['MasVnrArea'].mode()[0])

test_df['MSZoning'] = test_df['MSZoning'].fillna(test_df['MSZoning'].mode()[0])

test_df['BsmtHalfBath'] = test_df['BsmtHalfBath'].fillna(test_df['BsmtHalfBath'].mode()[0])

test_df['Utilities'] = test_df['Utilities'].fillna(test_df['Utilities'].mode()[0])

test_df['Functional'] = test_df['Functional'].fillna(test_df['Functional'].mode()[0])

test_df['BsmtFullBath'] = test_df['BsmtFullBath'].fillna(test_df['BsmtFullBath'].mode()[0])

test_df['BsmtFinSF2'] = test_df['BsmtFinSF2'].fillna(test_df['BsmtFinSF2'].mode()[0])

test_df['BsmtFinSF1'] = test_df['BsmtFinSF1'].fillna(test_df['BsmtFinSF1'].mode()[0])

test_df['BsmtUnfSF'] = test_df['BsmtUnfSF'].fillna(test_df['BsmtUnfSF'].mode()[0])

test_df['TotalBsmtSF'] = test_df['TotalBsmtSF'].fillna(test_df['TotalBsmtSF'].mode()[0])

test_df['Exterior2nd'] = test_df['Exterior2nd'].fillna(test_df['Exterior2nd'].mode()[0])

test_df['SaleType'] = test_df['SaleType'].fillna(test_df['SaleType'].mode()[0])

test_df['Exterior1st'] = test_df['Exterior1st'].fillna(test_df['Exterior1st'].mode()[0])

test_df['KitchenQual'] = test_df['KitchenQual'].fillna(test_df['KitchenQual'].mode()[0])

test_df['GarageArea'] = test_df['GarageArea'].fillna(test_df['GarageArea'].mode()[0])

test_df['GarageCars'] = test_df['GarageCars'].fillna(test_df['GarageCars'].mode()[0])
test_df.isnull().sum().max()
frames = [train_df, test_df]

full_data = pd.concat(frames, axis=0)

full_data['SalePrice']
full_data = pd.get_dummies(full_data, drop_first=True)
full_data.head()
## Transformation

# full_data['SalePrice'] = np.log(full_data['SalePrice'])

full_data['GrLivArea'] = np.log(full_data['GrLivArea'])
full_data['HasBsmt'] = pd.Series(len(full_data['TotalBsmtSF']), index=full_data.index)

full_data['HasBsmt'] = 0

full_data.loc[full_data['TotalBsmtSF'] > 0, 'HasBsmt'] = 1

full_data.loc[full_data['HasBsmt']==1, 'TotalBsmtSF'] = np.log(full_data['TotalBsmtSF'])

# train['HasBsmt'].head(20)

full_data.drop(['HasBsmt'], axis=1, inplace=True)
full_data.head()
df_train = full_data.iloc[:1460, :]

df_test = full_data.iloc[1460:,:]
df_train.shape
df_test.shape
df_test['SalePrice'].isnull().sum()

# df_test['SalePrice'].head(50)
df_test.drop(['SalePrice'], axis=1, inplace=True)
df_test.shape
df_train['SalePrice']
y_train = df_train['SalePrice']

y_train.head()

y_train.shape
x_train = df_train.drop(['SalePrice'], axis=1)

x_train.head()
import xgboost as xgb

from sklearn.metrics import mean_squared_error
xg_reg = xgb.XGBRegressor()

xg_reg
from sklearn.model_selection import RandomizedSearchCV

#Hyperparameter optimization

max_depth = [1,3,5,7,9,11,13,15,17,19,21]

n_estimators = [100, 200, 300, 400, 500, 600, 700]

booster = ['gbtree', 'gblinear']

learning_rate = [0.01, 0.05, 0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8]

min_child_weight = [1,3,5,7,9,11,13,15,17,19,21]



hyperparameter_grid = {

    "max_depth" : max_depth,

    "n_estimators" : n_estimators,

    "booster" : booster,

    "learning_rate" : learning_rate,

    "min_child_weight" : min_child_weight

    

}
random_cv = RandomizedSearchCV(estimator = xg_reg, param_distributions=hyperparameter_grid,

            cv=5, n_iter=50,

            scoring = 'neg_mean_absolute_error',n_jobs = 4,

            verbose = 5, 

            return_train_score = True,

            random_state=42)
random_cv.fit(x_train, y_train)
random_cv.best_estimator_
xg_reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,

             importance_type='gain', interaction_constraints='',

             learning_rate=0.05, max_delta_step=0, max_depth=5,

             min_child_weight=5, missing=None, monotone_constraints='()',

             n_estimators=700, n_jobs=0, num_parallel_tree=1, random_state=0,

             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,

             tree_method='exact', validate_parameters=1, verbosity=None)
xg_reg.fit(x_train, y_train)
preds = xg_reg.predict(df_test)

preds
pred=pd.DataFrame(preds)

subdf = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

datasets=pd.concat([subdf['Id'],pred],axis=1)

datasets.columns=['Id','SalePrice']

datasets.to_csv('sample_submission2.csv',index=False)
foo = pd.read_csv('sample_submission2.csv')

foo