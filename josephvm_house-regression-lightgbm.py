# LGBM

# sklearn environment

# uses pandas, numpy, sklearn, and lightgbm
import pandas as pd

import numpy as np
train_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train_df.head()
test_df.head()
y_train = train_df['SalePrice']

train_df.drop(['SalePrice'], inplace=True, axis=1)
train_df.info()
drop = ['Id', 'Alley', 'PoolQC', 'MiscFeature', 'Fence', 'FireplaceQu']

train_df.drop(drop, axis=1, inplace=True)

test_df.drop(drop, axis=1, inplace=True)
# Can just run SimpleImputer over all of the columns;

#   with a separate one for numeric columns and one for categorical columns



# Import

from sklearn.impute import SimpleImputer
# Imputer for numeric columns

mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')



# Imputer for categorical columns

common_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
# found this function; really like it

numerical = train_df.select_dtypes(exclude=['object']).columns

categorical = train_df.select_dtypes(exclude=['float64', 'int64']).columns
# fit the imputers

mean_imputer.fit(train_df[numerical])

common_imputer.fit(train_df[categorical])
# transform the data - numeric

train_df[numerical] = mean_imputer.transform(train_df[numerical]);

test_df[numerical] = mean_imputer.transform(test_df[numerical]);



# transform the data - categorical

train_df[categorical] = common_imputer.transform(train_df[categorical]);

test_df[categorical] = common_imputer.transform(test_df[categorical]);
train_df.columns[train_df.isnull().any()]
test_df.columns[test_df.isnull().any()]
# Quick correlation with SalePrice

pd.concat([train_df, y_train], axis=1).corr()['SalePrice'].sort_values(ascending=False)
# These need encoding

categorical.to_numpy()
from sklearn.preprocessing import OneHotEncoder



ohe = OneHotEncoder(sparse=False, categories='auto')
# Get the categorical columns

train_cat = train_df[categorical]

test_cat = test_df[categorical]
# Apply the OHE

train_ohe = ohe.fit_transform(train_cat)

test_ohe = ohe.transform(test_cat)
test_ohe
# Drop the old columns

train_df.drop(categorical, axis=1, inplace=True)

test_df.drop(categorical, axis=1, inplace=True)
# Concatenate what is left with the ohe columns

train_ready = pd.concat([train_df, pd.DataFrame(train_ohe)], axis=1)

test_ready = pd.concat([test_df, pd.DataFrame(test_ohe)], axis=1)
# Quick correlation with SalePrice

pd.concat([train_ready, y_train], axis=1).corr()['SalePrice'].sort_values(ascending=False)



# some ohe categories are fairly strongly negatively correlated
# Grid search actually didn't help, so it's commented out and almost the default parameters are used
from lightgbm import LGBMRegressor
# Grid Search

"""from sklearn.model_selection import GridSearchCV

param_grid = [{'min_data_in_leaf': [100, 200, 400], 

               'num_leaves': [5, 10, 20],

               'boosting_type': ['gbdt', 'dart'],

               'n_estimators': [250, 500, 1000],

               'learning_rate': [0.1, 0.05, 0.01]

              }]

param_grid2 = [{'n_estimators': [250, 500, 1000, 2000],

               'learning_rate': [0.1, 0.05, 0.01, 0.001]

              }]



# Get it ready

grid_search = GridSearchCV(forest, param_grid2, cv=7, verbose=1, n_jobs=-1)"""
#grid_search.fit(train_ready, y_train)
#grid_search.best_params_
#forest = grid_search.best_estimator_
forest = LGBMRegressor(n_estimators=500, learning_rate = 0.01)
forest.fit(train_ready, y_train)
y_pred = forest.predict(test_ready)
# Read in sample csv

sample_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
sample_df['SalePrice'] = y_pred
# Write to a new csv

sample_df.to_csv('predictions.csv', index=False) # Be sure to not include the index
# LightGBM (no grid search) - 0.13334

# LightGBM (grid search) - nothing better
# Ensemble learning to get better results

# Mess with the imputer

# Feature engineering with the slew of columns