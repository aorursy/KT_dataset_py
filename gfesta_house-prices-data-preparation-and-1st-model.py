# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



house_prices_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

house_prices_train['set'] = 'train'

y = house_prices_train.SalePrice

house_prices_train.drop('SalePrice', axis=1, inplace=True)



house_prices_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

house_prices_test['set'] = 'test'



house_prices = pd.concat([house_prices_train, house_prices_test], axis=0, sort=False)
house_prices.info()
# MSSubClass is actually categorical but encoded as integer numbers

house_prices['MSSubClass'] = house_prices['MSSubClass'].astype('object')



categorical_features = house_prices.columns[house_prices.dtypes == 'object'].to_list()

categorical_features.remove('set')



numerical_features = list(set(house_prices.columns) - set(categorical_features))

numerical_features.remove('Id')
import missingno as msno



msno.bar(house_prices)
msno.matrix(house_prices)
house_prices.isnull().sum()[house_prices.isnull().sum() != 0]
# percent of missing values. Features with no missing values are omitted

round(house_prices[categorical_features].isnull().mean()[house_prices[categorical_features].isnull().sum() > 0]*100,1)
features_w_missing_values = house_prices.columns[house_prices.isnull().sum() != 0]

msno.matrix(house_prices[features_w_missing_values])
msno.heatmap(house_prices[features_w_missing_values])
from sklearn.impute import SimpleImputer

# 'No alley access' is coded as NA: set is to 'No'

# FireplaceQU is NA when there is no Fireplace: set is to 'No'

# If the Poolarea is 0 (no pool) the PoolQC is NA: set it to 'No'

# analogously for Fence and MiscFeature missing will be set to 'No'



none_imputer_features = ['FireplaceQu','Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtExposure', 'BsmtFinType2',

                       'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 

                       'MiscFeature']



mostfreq_imputer_features = list(set(categorical_features) - set(none_imputer_features))



no_imputer = SimpleImputer(strategy='constant', fill_value='None')

no_imputer.fit(house_prices.loc[house_prices.set == 'train', none_imputer_features])

house_prices[none_imputer_features] = no_imputer.transform(house_prices[none_imputer_features])



msno.matrix(house_prices[features_w_missing_values])
house_prices[mostfreq_imputer_features].isnull().sum()
mostfreq_imputer = SimpleImputer(strategy='most_frequent')

mostfreq_imputer.fit(house_prices.loc[house_prices.set == 'train', mostfreq_imputer_features])

house_prices[mostfreq_imputer_features] = mostfreq_imputer.transform(house_prices[mostfreq_imputer_features])
round(house_prices[numerical_features].isnull().mean()[house_prices[numerical_features].isnull().sum() > 0]*100,1)
msno.matrix(house_prices.loc[:,house_prices.isnull().sum() != 0])
msno.heatmap(house_prices.loc[:,house_prices.isnull().sum() != 0])
import matplotlib.pyplot as plt

# NAs MasVnrAre correspond to No Masonry: set to 0

# NAs GarageYrrBlt correnspond to no garage: set to 0 to set level zero without garage



zero_imputer_features = ['MasVnrArea', 'GarageYrBlt', 'BsmtHalfBath', 'BsmtFullBath', 'BsmtFinSF1', 'BsmtFinSF2', 

                        'GarageCars', 'GarageArea', 'TotalBsmtSF', 'BsmtUnfSF']

zero_imputer = SimpleImputer(strategy='constant', fill_value=0)

zero_imputer.fit(house_prices.loc[house_prices.set == 'train', zero_imputer_features])

house_prices[zero_imputer_features] = zero_imputer.transform(house_prices[zero_imputer_features])
house_prices.isnull().sum()[house_prices.isnull().sum() != 0]
n = 0

for column_name, column_data in house_prices[categorical_features].iteritems():

    n_c = len(column_data.unique())

    print("{}: {}, {}".format(column_name, n_c, column_data.unique()))

    n += n_c

print('Expected dimension of One Hot encoding: ', n)

print('Numerical features dimension: ', len(numerical_features))

print('Total dimension: ', n + len(numerical_features))
from sklearn.preprocessing import OneHotEncoder



oe = OneHotEncoder(sparse = False)

oe.fit(house_prices[categorical_features])

oe_data = oe.transform(house_prices[categorical_features])

oe_df = pd.DataFrame(data=oe_data, columns=oe.get_feature_names(input_features=categorical_features), index=house_prices.index)



house_prices_oe = house_prices.drop(categorical_features, axis=1)



house_prices_oe = pd.concat([house_prices_oe, oe_df], axis=1)



house_prices_train_oe = house_prices_oe[house_prices_oe.set == 'train'].drop('set', axis=1)

house_prices_test_oe = house_prices_oe[house_prices_oe.set == 'test'].drop('set', axis=1)

house_prices_oe.drop('set', axis=1, inplace=True)

print(house_prices_oe.shape)

print(house_prices_train_oe.shape)

print(house_prices_test_oe.shape)
from sklearn.impute import KNNImputer



house_prices_knn = house_prices_oe.copy(deep=True)

knn_imp = KNNImputer()

knn_imp.fit(house_prices_train_oe)

house_prices_train_oe.loc[:,:] = knn_imp.transform(house_prices_train_oe)

house_prices_test_oe.loc[:,:] = knn_imp.transform(house_prices_test_oe)



print(house_prices_oe.shape)

print(house_prices_train_oe.shape)

print(house_prices_test_oe.shape)
house_prices_train_oe.isnull().sum()[house_prices_train_oe.isnull().sum() != 0]
house_prices_test_oe.isnull().sum()[house_prices_test_oe.isnull().sum() != 0]
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(house_prices_train_oe.drop('Id', axis=1), y, test_size=0.25, random_state=123)



X_train.shape
import xgboost as xgb

from hyperopt import STATUS_OK

from sklearn.model_selection import cross_val_score, KFold



N_FOLDS = 10



# define objective to minimize

def objective(params, n_folds = N_FOLDS):

    estimator = xgb.XGBRegressor(**params)

    # Perform cross-validation: cv_results

    loss = np.sqrt(-cross_val_score(estimator, X=X_train, y=y_train, scoring='neg_root_mean_squared_error',

        cv = N_FOLDS, n_jobs=-1)).mean()

    return {'loss': loss, 'params': params, 'status': STATUS_OK}
from hyperopt import hp



hyperparameter_space = {

    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),

    'subsample': hp.uniform('subsample', 0.6, 1.0),

    'min_child_weight': hp.quniform('min_child_weight', 1, 7, 2),

    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),

    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),

    'max_depth': hp.randint('max_depth', 1,16),

    'gamma': hp.uniform('gamma', 0.1,0.4),

    'max_delta_step': hp.randint('max_delta_step',0,10),

    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2))

}
from hyperopt import Trials



bayes_trials = Trials()
import numpy as np

from hyperopt import fmin

from hyperopt import tpe



MAX_EVALS = 50



best = fmin(fn = objective, space = hyperparameter_space, algo = tpe.suggest, max_evals = MAX_EVALS,

           trials = bayes_trials, rstate = np.random.RandomState(50))
best
from sklearn.metrics import mean_squared_error



best['num_boost_round']=10000

best['early_stopping_rounds']=100

best['objective']='reg:squarederror'

best['n_jobs'] = -1



xgb_best = xgb.XGBRegressor(**best)



print(xgb_best)



xgb_best.fit(X_train, y_train)



print('Train RMSE:', np.sqrt(mean_squared_error(y_train, xgb_best.predict(X_train))))

print('Test RMSE:', np.sqrt(mean_squared_error(y_test, xgb_best.predict(X_test))))



xgb_best.fit(house_prices_train_oe.drop('Id', axis=1), y)
#print(house_prices_test_oe.info())

house_prices_test_oe['SalePrice'] = xgb_best.predict(house_prices_test_oe.drop('Id', axis=1))

house_prices_test_oe[['Id','SalePrice']].head()



house_prices_test_oe['Id'] = house_prices_test_oe['Id'].astype('int')

house_prices_test_oe[['Id','SalePrice']].to_csv('submission.csv', index=False)
house_prices_test_oe[['Id','SalePrice']].shape