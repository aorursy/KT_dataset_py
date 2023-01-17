# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import sys

import os

import warnings

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np # linear algebra 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

if not sys.warnoptions:

    warnings.simplefilter("ignore")
# Read the data

df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')

num_columns = df.select_dtypes(include=np.number)

cat_columns = df.select_dtypes(exclude=np.number)





print(f'We have {len(df.columns)} columns, {len(num_columns.columns)} of which are numerical.')
df.SalePrice.describe()
sns.distplot(df.SalePrice)
#features_to_analyse = ['MSZoning', 'ExterQual']

features_to_analyse = cat_columns

for f in features_to_analyse:

    sns.boxplot(f, 'SalePrice', data=df)

    plt.show()
X_test_full = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



test_ID = X_test_full['Id']

X_test_full.drop(['Id'], axis=1, inplace=True)



# Separate target from predictors

y = df.SalePrice              

X = df.drop(['SalePrice'], axis=1)



# Break off validation set from training data

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)



# Select categorical columns with relatively low cardinality (convenient but arbitrary)

low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 

                        X_train_full[cname].dtype == "object"]



# Select numeric columns

numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = low_cardinality_cols + numeric_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()



# One-hot encode the data

X_train = pd.get_dummies(X_train)

X_valid = pd.get_dummies(X_valid)

X_test = pd.get_dummies(X_test)

X_train, X_valid = X_train.align(X_valid, join='left', axis=1)

X_train, X_test = X_train.align(X_test, join='left', axis=1)
print(X_train.shape)

print(y_train.shape)



print(X_valid.shape)

print(y_valid.shape)



print(X_test.shape)
# model tuning



from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

import time



# A parameter grid for XGBoost

params = {

    'n_estimators':[500],

    'min_child_weight':[4,5], 

    'gamma':[i/10.0 for i in range(3,6)],  

    'subsample':[i/10.0 for i in range(6,11)],

    'colsample_bytree':[i/10.0 for i in range(6,11)], 

    'max_depth': [2,3,4,6,7],

    'objective': ['reg:squarederror', 'reg:tweedie'],

    'booster': ['gbtree', 'gblinear'],

    'eval_metric': ['rmse'],

    'eta': [i/10.0 for i in range(3,6)],

}



reg = XGBRegressor(nthread=-1)



# run randomized search

n_iter_search = 100

random_search = RandomizedSearchCV(reg, param_distributions=params,

                                   n_iter=n_iter_search, cv=5, iid=False, scoring='neg_mean_squared_error')



start = time.time()

random_search.fit(X_train, y_train)

print("RandomizedSearchCV took %.2f seconds for %d candidates"

      " parameter settings." % ((time.time() - start), n_iter_search))
best_regressor = random_search.best_estimator_
from sklearn.metrics import mean_absolute_error



# Get predictions

y_pred = best_regressor.predict(X_valid)
# Calculate MAE

rmse_pred = mean_absolute_error(y_valid, y_pred) 



print("Root Mean Absolute Error:" , np.sqrt(rmse_pred))
# Get predictions

y_pred_test = best_regressor.predict(X_test)
sub = pd.DataFrame()

sub['Id'] = test_ID

sub['SalePrice'] = y_pred_test

sub.to_csv('submission.csv',index=False)