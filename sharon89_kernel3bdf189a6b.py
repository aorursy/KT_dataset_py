import pandas as pd

%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)

mpl.rc('xtick', labelsize=12)

mpl.rc('ytick', labelsize=12)

import sklearn

from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, PowerTransformer, OneHotEncoder, StandardScaler
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
print('train:', train.head())

print('test:', test.head())
train.drop(['Id', 'Utilities', 'Street', 'LowQualFinSF', 'PoolArea'], axis=1, inplace=True)

test.drop(['Id', 'Utilities', 'Street', 'LowQualFinSF', 'PoolArea'], axis=1, inplace=True)
train.describe()
train.info()
%matplotlib inline

import matplotlib.pyplot as plt

train.hist(bins=50, figsize=(36, 24))

plt.show()
train.drop(['3SsnPorch', 'EnclosedPorch', 'MiscVal', 'ScreenPorch', 'BsmtFinSF2'], axis=1, inplace=True)

test.drop(['3SsnPorch', 'EnclosedPorch', 'MiscVal', 'ScreenPorch', 'BsmtFinSF2'], axis=1, inplace=True)
train.info()
print(train['OverallQual'].value_counts())

print(train.shape)
# from sklearn.model_selection import StratifiedShuffleSplit



# split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# for train_index, test_index in split.split(train, train['OverallQual']):

#     train_set = train.loc[train_index]

#     test_set = train.loc[test_index]



from sklearn.model_selection import train_test_split



train_set, test_set = train_test_split(train, test_size=0.2, random_state=42)
train_set['OverallQual'].value_counts() / len(train_set)
test_set['OverallQual'].value_counts() / len(test_set)
train_set_copy = train_set.copy()

print(train_set_copy)
corr_matrix = train_set.corr()
corr_matrix['SalePrice'].sort_values(ascending=False)
from pandas.plotting import scatter_matrix



attributes = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',\

               'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', \

               'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'Fireplaces', \

               'BsmtFinSF1', 'LotFrontage', 'WoodDeckSF', '2ndFlrSF', 'OpenPorchSF']

scatter_matrix(train_set[attributes], figsize=(48, 32))

plt.show()
train_missing = train_set.isnull().sum().sort_values(ascending=False)

train_missing.head(20)
train_set['MiscFeature'].fillna('None', inplace=True)

train_set['Fence'].fillna('No Fence', inplace=True)

train_set['PoolQC'].fillna('No Pool', inplace=True)

train_set['Alley'].fillna('No alley access', inplace=True)

train_set['FireplaceQu'].fillna('No Fireplace', inplace=True)

train_set['LotFrontage'].fillna(train['LotFrontage'].median(), inplace=True)

train_set['GarageCond'].fillna('No Garage', inplace=True)

train_set['GarageType'].fillna('No Garage', inplace=True)

train_set['GarageYrBlt'].fillna(round(train['GarageYrBlt'].median(), 1), inplace=True)

train_set['GarageFinish'].fillna('No Garage', inplace=True)

train_set['GarageQual'].fillna('No Garage', inplace=True)

train_set['BsmtExposure'].fillna('No Basement', inplace=True)

train_set['BsmtFinType2'].fillna('No Basement', inplace=True)

train_set['BsmtFinType1'].fillna('No Basement', inplace=True)

train_set['BsmtCond'].fillna('No Basement', inplace=True)

train_set['BsmtQual'].fillna('No Basement', inplace=True)

train_set['MasVnrArea'].fillna(0.0, inplace=True)

train_set['MasVnrType'].fillna('None', inplace=True)

train_set['Electrical'].fillna('Mixed', inplace=True)
train_set.info()
columns = train_set.columns

print(columns)
numeric = ['LotFrontage', 

           'LotArea',  

           'YearBuilt', 

           'YearRemodAdd',

           'MasVnrArea', 

           'BsmtFinSF1', 

           'BsmtFinSF2', 

           'BsmtUnfSF', 

           'TotalBsmtSF',

           '1stFlrSF', 

           '2ndFlrSF', 

           'GrLivArea',  

           'GarageArea', 

           'WoodDeckSF', 

           'OpenPorchSF',

           'EnclosedPorch', 

           '3SsnPorch', 

           'ScreenPorch', 

           'MiscVal', 

           'GarageYrBlt',

           'SalePrice']
categorical = list(set(columns) - set(numeric))

print(categorical)
# from sklearn import preprocessing

# le = preprocessing.LabelEncoder()

# for col in categorical:

#     le.fit(train_set[[col]])

#     le.transform(train_set[[col]])





from sklearn.preprocessing import OrdinalEncoder



enc = OrdinalEncoder()

enc.fit(train_set[categorical])

train_set[categorical] = enc.transform(train_set[categorical])
train_set.head()
from sklearn.ensemble import RandomForestRegressor
rf_clf = RandomForestRegressor(random_state=42, n_estimators=500, max_depth=3, criterion='mse')
rf_clf.fit(train_set.drop(['SalePrice'], axis=1), train_set['SalePrice'])
from sklearn.model_selection import KFold, cross_val_score



cross_val_score(rf_clf, train_set.drop(['SalePrice'], axis=1), train_set['SalePrice'], cv=5)
import numpy as np



importances = rf_clf.feature_importances_

std = np.std([tree.feature_importances_ for tree in rf_clf.estimators_], axis=0)

indices = np.argsort(importances)[::-1]



print("Feature ranking:")



for f in range(train_set.shape[1] - 1):

    print("%d. %s -- feature %d (%f)" % (f + 1, train_set.columns[indices[f]], indices[f], importances[indices[f]]))
columns_to_models = list(train_set.columns[indices[:40]])

columns_to_models.append('SalePrice')

print(columns_to_models)
train_set = train_set_copy[columns_to_models]

test_set = test_set[columns_to_models]

print(train_set.isnull().sum().sort_values(ascending=False))

print(test_set.isnull().sum().sort_values(ascending=False))
test = test[columns_to_models[0:-1]]

print(test.isnull().sum().sort_values(ascending=False))

test['Fence'].fillna('No Fence', inplace=True)

test['PoolQC'].fillna('No Pool', inplace=True)

test['FireplaceQu'].fillna('No Fireplace', inplace=True)

test['LotFrontage'].fillna(train['LotFrontage'].median(), inplace=True)

test['GarageType'].fillna('No Garage', inplace=True)

test['GarageYrBlt'].fillna(round(test['GarageYrBlt'].median(), 1), inplace=True)

test['BsmtFinType2'].fillna('No Basement', inplace=True)

test['BsmtFinType1'].fillna('No Basement', inplace=True)

test['BsmtQual'].fillna('No Basement', inplace=True)

test['MasVnrArea'].fillna(0.0, inplace=True)

test['MSZoning'].fillna('RL', inplace=True)

test['BsmtUnfSF'].fillna(0.0, inplace=True)

test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].median(), inplace=True)

test['BsmtFinSF1'].fillna(0.0, inplace=True)

test['GarageCars'].fillna(2.0, inplace=True)

test['GarageArea'].fillna(0.0, inplace=True)

test['KitchenQual'].fillna('TA', inplace=True)

test['Exterior1st'].fillna('VinylSd', inplace=True)
train_set['PoolQC'].fillna('No Pool', inplace=True)

train_set['LotFrontage'].fillna(train['LotFrontage'].median(), inplace=True)

train_set['GarageType'].fillna('No Garage', inplace=True)

# train_set['GarageFinish'].fillna('No Garage', inplace=True)

train_set['GarageYrBlt'].fillna(round(train['GarageYrBlt'].median(), 1), inplace=True)

# train_set['GarageQual'].fillna('No Garage', inplace=True)

train_set['BsmtFinType2'].fillna('No Basement', inplace=True)

# train_set['BsmtExposure'].fillna('No Basement', inplace=True)

train_set['BsmtQual'].fillna('No Basement', inplace=True)

train_set['BsmtFinType1'].fillna('No Basement', inplace=True)

train_set['MasVnrArea'].fillna(0.0, inplace=True)

train_set['Fence'].fillna('No Fence', inplace=True)

train_set['FireplaceQu'].fillna('No Fireplace', inplace=True)
test_set['PoolQC'].fillna('No Pool', inplace=True)

test_set['LotFrontage'].fillna(train['LotFrontage'].median(), inplace=True)

test_set['GarageType'].fillna('No Garage', inplace=True)

# test_set['GarageFinish'].fillna('No Garage', inplace=True)

test_set['GarageYrBlt'].fillna(round(train['GarageYrBlt'].median(), 1), inplace=True)

# test_set['GarageQual'].fillna('No Garage', inplace=True)

test_set['BsmtFinType2'].fillna('No Basement', inplace=True)

# test_set['BsmtExposure'].fillna('No Basement', inplace=True)

test_set['BsmtQual'].fillna('No Basement', inplace=True)

test_set['BsmtFinType1'].fillna('No Basement', inplace=True)

test_set['MasVnrArea'].fillna(0.0, inplace=True)

test_set['Fence'].fillna('No Fence', inplace=True)

test_set['FireplaceQu'].fillna('No Fireplace', inplace=True)
# train_set['BsmtQual'].fillna('No Basement', inplace=True)

# test_set['BsmtQual'].fillna('No Basement', inplace=True)

train_set_original = train_set.copy()

test_set_original = test_set.copy()
test_set.columns
# numeric_test = ['LotArea',  

#           'BsmtFinSF1', 

#           'TotalBsmtSF',

#           '1stFlrSF', 

#           '2ndFlrSF', 

#           'GrLivArea',  

#           'SalePrice']

numeric_test = ['LotFrontage', 

           'LotArea',  

           'YearBuilt', 

           'YearRemodAdd',

           'MasVnrArea', 

           'BsmtFinSF1', 

           'BsmtFinSF2', 

           'BsmtUnfSF', 

           'TotalBsmtSF',

           '1stFlrSF', 

           '2ndFlrSF', 

           'GrLivArea',  

           'GarageArea', 

           'WoodDeckSF', 

           'OpenPorchSF',

           'EnclosedPorch', 

           '3SsnPorch', 

           'ScreenPorch', 

           'MiscVal', 

           'GarageYrBlt',

           'SalePrice']

categorical_test = list(set(test_set.columns) - set(numeric_test))

print(categorical_test)
data = pd.concat([train_set[categorical_test], test_set[categorical_test], test[categorical_test]])
enc_test = OrdinalEncoder()

enc_test.fit(data)

train_set[categorical_test] = enc_test.transform(train_set[categorical_test])

test_set[categorical_test] = enc_test.transform(test_set[categorical_test])

print(train_set.head())

print(test_set.head())
test[categorical_test] = enc_test.transform(test[categorical_test])
X_train = train_set.loc[:, train_set.columns != 'SalePrice']

y_train = train_set.loc[:, 'SalePrice']

print(y_train)
X_test = test_set.loc[:, test_set.columns != 'SalePrice']

y_test = test_set.loc[:, 'SalePrice']

print('X_test:', X_test, 'y_test:', y_test) 
print('train_set_original:', train_set_original, 'test_set_original:', test_set_original)
X_train_original = train_set_original.loc[:, train_set_original.columns != 'SalePrice']

y_train_original = train_set_original.loc[:, 'SalePrice']

X_test_original = test_set_original.loc[:, test_set_original.columns != 'SalePrice']

y_test_original = test_set_original.loc[:, 'SalePrice']
from sklearn.preprocessing import StandardScaler



#np.concatenate((a,b),axis=1)



scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

test = scaler.transform(test)
import xgboost as xgb

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error



kfolds = KFold(n_splits=10, shuffle=True, random_state=42)
xgb_reg = xgb.XGBRegressor(learning_rate=0.01, n_estimators=100, 

                           n_jobs=-1, booster='gbtree', random_state=42, subsample=0.5)

xgb_param_grid = {"learning_rate":[0.11, 0.12, 0.13],

                  "n_estimators":[80, 90, 95, 100, 110],

                  "subsample":[0.6, 0.7, 0.8, 0.9],

                  "max_depth":[3, 4, 5, 6]

                  }

                  

grid_search = GridSearchCV(xgb_reg, param_grid=xgb_param_grid, cv=kfolds, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)

grid_search.fit(X_train, y_train) #, eval_set = [(X_valid_new, y_valid_new)], early_stopping_rounds=2)

xgb_reg = grid_search.best_estimator_

print(grid_search.best_params_)



y_train_pred = xgb_reg.predict(X_train)

print('-' * 10 + 'XGB' + '-' * 10)

print('R square Accuracy for train: ', r2_score(y_train, y_train_pred))

print('Mean Absolute Error for train: ', mean_absolute_error(y_train, y_train_pred))

print('Mean Squared Error for train: ', mean_squared_error(y_train, y_train_pred))



y_pred = xgb_reg.predict(X_test)



print('-' * 10 + 'XGB' + '-' * 10)

print('R square Accuracy: ', r2_score(y_test, y_pred))

print('Mean Absolute Error: ', mean_absolute_error(y_test, y_pred))

print('Mean Squared Error: ', mean_squared_error(y_test, y_pred))
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import expon, reciprocal



# see https://docs.scipy.org/doc/scipy/reference/stats.html

# for `expon()` and `reciprocal()` documentation and more probability distribution functions.



# Note: gamma is ignored when kernel is "linear"

xgb_reg = xgb.XGBRegressor(learning_rate=0.01, n_estimators=100, 

                           n_jobs=-1, booster='gbtree', random_state=42, subsample=0.5)

param_distribs = {

        "learning_rate": np.linspace(0.01, 0.2, 100),

        "n_estimators": range(10, 100, 25),

        "subsample": np.linspace(0.5, 0.7, 20),

        "max_depth": range(3, 10, 1),

        "lambda": np.linspace(1, 5, 100)

        #"lambda": expon(scale=1.0)

    }



rnd_search = RandomizedSearchCV(xgb_reg, param_distributions=param_distribs, cv=kfolds, scoring="neg_mean_squared_error", n_jobs=-1, verbose=2)

rnd_search.fit(X_train, y_train)



xgb_reg1 = rnd_search.best_estimator_

print(rnd_search.best_params_)



y_train_pred = xgb_reg1.predict(X_train)

print('-' * 10 + 'XGB' + '-' * 10)

print('R square Accuracy for train: ', r2_score(y_train, y_train_pred))

print('Mean Absolute Error for train: ', mean_absolute_error(y_train, y_train_pred))

print('Mean Squared Error for train: ', mean_squared_error(y_train, y_train_pred))



y_pred = xgb_reg1.predict(X_test)

print('-' * 10 + 'XGB' + '-' * 10)

print('R square Accuracy: ', r2_score(y_test, y_pred))

print('Mean Absolute Error: ', mean_absolute_error(y_test, y_pred))

print('Mean Squared Error: ', mean_squared_error(y_test, y_pred))
import lightgbm as lgbm
lgb_reg = lgbm.LGBMRegressor(objective='regression', learning_rate=0.01, n_estimators=100, n_jobs=-1, subsample=0.5)

lgb_param_grid = {'max_depth': [2, 3, 4, 5, 6],

                  'learning_rate': [0.01, 0.02, 0.03, 0.04],

                  "n_estimators":[700, 800, 900],

                  'bagging_fraction': [0.1, 0.2, 0.3],

                  }

                  

grid_search = GridSearchCV(lgb_reg, param_grid=lgb_param_grid, cv=kfolds, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)

grid_search.fit(X_train, y_train) #, eval_set = [(X_valid_new, y_valid_new)], early_stopping_rounds=2)



lgb_reg = grid_search.best_estimator_

print(grid_search.best_params_)



y_train_pred = lgb_reg.predict(X_train)

print('-' * 10 + 'LGBM' + '-' * 10)

print('R square Accuracy for train: ', r2_score(y_train, y_train_pred))

print('Mean Absolute Error for train: ', mean_absolute_error(y_train, y_train_pred))

print('Mean Squared Error for train: ', mean_squared_error(y_train, y_train_pred))





y_pred = lgb_reg.predict(X_test)



print('-' * 10 + 'LGBM' + '-' * 10)

print('R square Accuracy: ', r2_score(y_test, y_pred))

print('Mean Absolute Error: ', mean_absolute_error(y_test, y_pred))

print('Mean Squared Error: ', mean_squared_error(y_test, y_pred))
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import expon, reciprocal



# see https://docs.scipy.org/doc/scipy/reference/stats.html

# for `expon()` and `reciprocal()` documentation and more probability distribution functions.



# Note: gamma is ignored when kernel is "linear"

lgb_reg = lgbm.LGBMRegressor(objective='regression', learning_rate=0.01, n_estimators=100, n_jobs=-1, subsample=0.5)



param_distribs = {

        "learning_rate": np.linspace(0.001, 0.5, 100),

        "n_estimators": range(50, 500, 25),

        "bagging_fraction": np.linspace(0.1, 0.8, 20),

        "max_depth": range(2, 7, 1)

    }



rnd_search = RandomizedSearchCV(lgb_reg, param_distributions=param_distribs, cv=kfolds, scoring="neg_mean_squared_error", n_jobs=-1, verbose=2)

rnd_search.fit(X_train, y_train)



lgb_reg1 = rnd_search.best_estimator_

print(rnd_search.best_params_)



y_train_pred = lgb_reg1.predict(X_train)

print('-' * 10 + 'LGBM' + '-' * 10)

print('R square Accuracy for train: ', r2_score(y_train, y_train_pred))

print('Mean Absolute Error for train: ', mean_absolute_error(y_train, y_train_pred))

print('Mean Squared Error for train: ', mean_squared_error(y_train, y_train_pred))



y_pred = lgb_reg1.predict(X_test)

print('-' * 10 + 'LGBM' + '-' * 10)

print('R square Accuracy: ', r2_score(y_test, y_pred))

print('Mean Absolute Error: ', mean_absolute_error(y_test, y_pred))

print('Mean Squared Error: ', mean_squared_error(y_test, y_pred))
# import catboost as cat
# cat_reg = cat.CatBoostRegressor(cat_features=categorical_test,learning_rate=0.5, loss_function='R2',

#                                 objective='RMSE', random_state=42, n_estimators=100, max_depth=3)

# cat_param_grid = {"learning_rate":[0.1, 0.01, 0.05],

#                   "n_estimators":[200, 250, 500],

#                   "l2_leaf_reg":[3, 4, 5],

#                   "max_depth":[5, 6, 7]

#                   }

# grid_search = GridSearchCV(cat_reg, param_grid=cat_param_grid, cv=kfolds, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)

# grid_search.fit(X_train_original, y_train_original)



# cat_reg = grid_search.best_estimator_

# print(grid_search.best_params_)



# y_train_pred = cat_reg.predict(X_train_original)

# print('-' * 10 + 'CatBoost' + '-' * 10)

# print('R square Accuracy for train: ', r2_score(y_train_original, y_train_pred))

# print('Mean Absolute Error for train: ', mean_absolute_error(y_train_original, y_train_pred))

# print('Mean Squared Error for train: ', mean_squared_error(y_train_original, y_train_pred))





# y_pred = cat_reg.predict(X_test_original)



# print('-' * 10 + 'CatBoost' + '-' * 10)

# print('R square Accuracy: ', r2_score(y_test_original, y_pred))

# print('Mean Absolute Error: ', mean_absolute_error(y_test_original, y_pred))

# print('Mean Squared Error: ', mean_squared_error(y_test_original, y_pred))

# from sklearn.model_selection import RandomizedSearchCV

# from scipy.stats import expon, reciprocal



# # see https://docs.scipy.org/doc/scipy/reference/stats.html

# # for `expon()` and `reciprocal()` documentation and more probability distribution functions.



# # Note: gamma is ignored when kernel is "linear"

# cat_reg = cat.CatBoostRegressor(cat_features=categorical_test,learning_rate=0.5, loss_function='R2',

#                                 objective='RMSE', random_state=42, n_estimators=100, max_depth=3)



# param_distribs = {

#         "learning_rate": np.linspace(0.001, 0.5, 100),

#         "n_estimators": range(50, 500, 25),

#         "l2_leaf_reg": range(2, 10, 1),

#         "max_depth": range(2, 7, 1)

#     }



# rnd_search = RandomizedSearchCV(cat_reg, param_distributions=param_distribs, cv=kfolds, scoring="neg_mean_squared_error", n_jobs=-1, verbose=2)

# rnd_search.fit(X_train_original, y_train_original)



# cat_reg1 = rnd_search.best_estimator_

# print(rnd_search.best_params_)



# y_train_pred = cat_reg1.predict(X_train_original)

# print('-' * 10 + 'CatBoost' + '-' * 10)

# print('R square Accuracy for train: ', r2_score(y_train_original, y_train_pred))

# print('Mean Absolute Error for train: ', mean_absolute_error(y_train_original, y_train_pred))

# print('Mean Squared Error for train: ', mean_squared_error(y_train_original, y_train_pred))



# y_pred = cat_reg1.predict(X_test_original)

# print('-' * 10 + 'CatBoost' + '-' * 10)

# print('R square Accuracy: ', r2_score(y_test_original, y_pred))

# print('Mean Absolute Error: ', mean_absolute_error(y_test_original, y_pred))

# print('Mean Squared Error: ', mean_squared_error(y_test_original, y_pred))
from sklearn.ensemble import GradientBoostingRegressor
gb_reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.01, random_state=42, subsample=0.5)

gbdt_param_grid = {"learning_rate":[0.1, 0.01, 0.001],

                   "n_estimators":[250, 500, 750, 1000],

                   "subsample":[0.3, 0.4, 0.5, 0.6],

                   "max_depth":[4, 5, 6, 7]

                  }



                  

grid_search = GridSearchCV(gb_reg, param_grid=gbdt_param_grid, cv=kfolds, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)

grid_search.fit(X_train, y_train)



gb_reg = grid_search.best_estimator_

print(grid_search.best_params_)



y_train_pred = gb_reg.predict(X_train)

print('-' * 10 + 'GBM' + '-' * 10)

print('R square Accuracy for train: ', r2_score(y_train, y_train_pred))

print('Mean Absolute Error for train: ', mean_absolute_error(y_train, y_train_pred))

print('Mean Squared Error for train: ', mean_squared_error(y_train, y_train_pred))



y_pred = gb_reg.predict(X_test)

print('-' * 10 + 'GBM' + '-' * 10)

print('R square Accuracy: ', r2_score(y_test, y_pred))

print('Mean Absolute Error: ', mean_absolute_error(y_test, y_pred))

print('Mean Squared Error: ', mean_squared_error(y_test, y_pred))
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import expon, reciprocal



# see https://docs.scipy.org/doc/scipy/reference/stats.html

# for `expon()` and `reciprocal()` documentation and more probability distribution functions.



# Note: gamma is ignored when kernel is "linear"

gb_reg = GradientBoostingRegressor()



param_distribs = {

        "learning_rate": np.linspace(0.001, 0.05, 100),

        "n_estimators": range(50, 1000, 25),

        "subsample": np.linspace(0.3, 0.9, 20),

        "max_depth": range(2, 7, 1)

    }



rnd_search = RandomizedSearchCV(gb_reg, param_distributions=param_distribs, cv=kfolds, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)

rnd_search.fit(X_train, y_train)



gb_reg1 = rnd_search.best_estimator_

print(rnd_search.best_params_)



y_train_pred = gb_reg1.predict(X_train)

print('-' * 10 + 'GBM' + '-' * 10)

print('R square Accuracy for train: ', r2_score(y_train, y_train_pred))

print('Mean Absolute Error for train: ', mean_absolute_error(y_train, y_train_pred))

print('Mean Squared Error for train: ', mean_squared_error(y_train, y_train_pred))



y_pred = gb_reg1.predict(X_test)

print('-' * 10 + 'GBM' + '-' * 10)

print('R square Accuracy: ', r2_score(y_test, y_pred))

print('Mean Absolute Error: ', mean_absolute_error(y_test, y_pred))

print('Mean Squared Error: ', mean_squared_error(y_test, y_pred))
from sklearn.svm import SVR 



param_grid = [

        {'kernel': ['linear'], 'C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},

        {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0],

         'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},

    ]



svm_reg = SVR()

grid_search = GridSearchCV(svm_reg, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)

grid_search.fit(X_train, y_train)

svr_reg = grid_search.best_estimator_

print(grid_search.best_params_)



y_train_pred = svr_reg.predict(X_train)

print('-' * 10 + 'SVR' + '-' * 10)

print('R square Accuracy for train: ', r2_score(y_train, y_train_pred))

print('Mean Absolute Error for train: ', mean_absolute_error(y_train, y_train_pred))

print('Mean Squared Error for train: ', mean_squared_error(y_train, y_train_pred))



y_pred = svr_reg.predict(X_test)

print('-' * 10 + 'SVR' + '-' * 10)

print('R square Accuracy: ', r2_score(y_test, y_pred))

print('Mean Absolute Error: ', mean_absolute_error(y_test, y_pred))

print('Mean Squared Error: ', mean_squared_error(y_test, y_pred))
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import expon, reciprocal



# see https://docs.scipy.org/doc/scipy/reference/stats.html

# for `expon()` and `reciprocal()` documentation and more probability distribution functions.



# Note: gamma is ignored when kernel is "linear"

param_distribs = {

        'kernel': ['linear', 'rbf'],

        'C': reciprocal(20, 5000),

        'gamma': expon(scale=1.0),

    }



svm_reg = SVR()

rnd_search = RandomizedSearchCV(svm_reg, param_distributions=param_distribs,

                                n_iter=50, cv=kfolds, scoring='neg_mean_squared_error',

                                verbose=1, random_state=42)

rnd_search.fit(X_train, y_train)

svr_reg1 = rnd_search.best_estimator_

print(rnd_search.best_params_)



y_train_pred = svr_reg1.predict(X_train)

print('-' * 10 + 'SVR' + '-' * 10)

print('R square Accuracy for train: ', r2_score(y_train, y_train_pred))

print('Mean Absolute Error for train: ', mean_absolute_error(y_train, y_train_pred))

print('Mean Squared Error for train: ', mean_squared_error(y_train, y_train_pred))



y_pred = svr_reg1.predict(X_test)

print('-' * 10 + 'SVR' + '-' * 10)

print('R square Accuracy: ', r2_score(y_test, y_pred))

print('Mean Absolute Error: ', mean_absolute_error(y_test, y_pred))

print('Mean Squared Error: ', mean_squared_error(y_test, y_pred))
from sklearn.linear_model import LinearRegression

# from sklearn.model_selection import cross_val_score



lin_reg = LinearRegression().fit(X_train, y_train)

y_train_pred = lin_reg.predict(X_train)



print('-' * 10 + 'Linear Regression' + '-' * 10)

print('R square Accuracy for train: ', r2_score(y_train, y_train_pred))

print('Mean Absolute Error for train: ', mean_absolute_error(y_train, y_train_pred))

print('Mean Squared Error for train: ', mean_squared_error(y_train, y_train_pred))



y_pred = lin_reg.predict(X_test)

print('-' * 10 + 'Linear Regression' + '-' * 10)

print('R square Accuracy: ', r2_score(y_test, y_pred))

print('Mean Absolute Error: ', mean_absolute_error(y_test, y_pred))

print('Mean Squared Error: ', mean_squared_error(y_test, y_pred))
lin_reg.coef_
X_train_original.columns
from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression



for degree in range(2, 3):

    polybig_features = PolynomialFeatures(degree=degree, include_bias=False)

    lin_reg = LinearRegression()

    polynomial_regression = Pipeline([

            ("poly_features", polybig_features),

            ("lin_reg", lin_reg),

        ])

    polynomial_regression.fit(X_train, y_train)

    y_train_pred = polynomial_regression.predict(X_train)

    print('degree:', degree)

    print('-' * 10 + 'Polynomial Regression' + '-' * 10)

    print('R square Accuracy for train: ', r2_score(y_train, y_train_pred))

    print('Mean Absolute Error for train: ', mean_absolute_error(y_train, y_train_pred))

    print('Mean Squared Error for train: ', mean_squared_error(y_train, y_train_pred))



    y_pred = polynomial_regression.predict(X_test)

    print('-' * 10 + 'Polynomial Regression' + '-' * 10)

    print('R square Accuracy: ', r2_score(y_test, y_pred))

    print('Mean Absolute Error: ', mean_absolute_error(y_test, y_pred))

    print('Mean Squared Error: ', mean_squared_error(y_test, y_pred))
from sklearn.linear_model import Ridge
param_grid = [

        {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2]},

    ]



ridge_reg = Ridge(alpha=1, random_state=42)

grid_search = GridSearchCV(ridge_reg, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)

grid_search.fit(X_train, y_train)

ridge_reg = grid_search.best_estimator_

print(grid_search.best_params_)



y_train_pred = ridge_reg.predict(X_train)

print('-' * 10 + 'Ridge' + '-' * 10)

print('R square Accuracy for train: ', r2_score(y_train, y_train_pred))

print('Mean Absolute Error for train: ', mean_absolute_error(y_train, y_train_pred))

print('Mean Squared Error for train: ', mean_squared_error(y_train, y_train_pred))



y_pred = ridge_reg.predict(X_test)

print('-' * 10 + 'Ridge' + '-' * 10)

print('R square Accuracy: ', r2_score(y_test, y_pred))

print('Mean Absolute Error: ', mean_absolute_error(y_test, y_pred))

print('Mean Squared Error: ', mean_squared_error(y_test, y_pred))
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import expon, reciprocal



# see https://docs.scipy.org/doc/scipy/reference/stats.html

# for `expon()` and `reciprocal()` documentation and more probability distribution functions.



# Note: gamma is ignored when kernel is "linear"

ridge_reg = Ridge(alpha=1, random_state=42)



param_distribs = {

        'alpha': expon(scale=1.0)

    }



rnd_search = RandomizedSearchCV(ridge_reg, param_distributions=param_distribs, cv=kfolds, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)

rnd_search.fit(X_train, y_train)



ridge_reg1 = rnd_search.best_estimator_

print(rnd_search.best_params_)



y_train_pred = ridge_reg1.predict(X_train)

print('-' * 10 + 'Ridge' + '-' * 10)

print('R square Accuracy for train: ', r2_score(y_train, y_train_pred))

print('Mean Absolute Error for train: ', mean_absolute_error(y_train, y_train_pred))

print('Mean Squared Error for train: ', mean_squared_error(y_train, y_train_pred))



y_pred = ridge_reg1.predict(X_test)

print('-' * 10 + 'Ridge' + '-' * 10)

print('R square Accuracy: ', r2_score(y_test, y_pred))

print('Mean Absolute Error: ', mean_absolute_error(y_test, y_pred))

print('Mean Squared Error: ', mean_squared_error(y_test, y_pred))
from sklearn.linear_model import Lasso
param_grid = [

        {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 2, 5, 10, 20]},

    ]



lasso_reg = Lasso(alpha=1, random_state=42)

grid_search = GridSearchCV(lasso_reg, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)

grid_search.fit(X_train, y_train)

lasso_reg = grid_search.best_estimator_

print(grid_search.best_params_)



y_train_pred = lasso_reg.predict(X_train)

print('-' * 10 + 'Lasso' + '-' * 10)

print('R square Accuracy for train: ', r2_score(y_train, y_train_pred))

print('Mean Absolute Error for train: ', mean_absolute_error(y_train, y_train_pred))

print('Mean Squared Error for train: ', mean_squared_error(y_train, y_train_pred))



y_pred = lasso_reg.predict(X_test)

print('-' * 10 + 'Lasso' + '-' * 10)

print('R square Accuracy: ', r2_score(y_test, y_pred))

print('Mean Absolute Error: ', mean_absolute_error(y_test, y_pred))

print('Mean Squared Error: ', mean_squared_error(y_test, y_pred))
lasso_reg.coef_
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import expon, reciprocal



# see https://docs.scipy.org/doc/scipy/reference/stats.html

# for `expon()` and `reciprocal()` documentation and more probability distribution functions.



# Note: gamma is ignored when kernel is "linear"

lasso_reg = Lasso(alpha=1, random_state=42)



param_distribs = {

        'alpha': expon(scale=1.0)

    }



rnd_search = RandomizedSearchCV(lasso_reg, param_distributions=param_distribs, cv=kfolds, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)

rnd_search.fit(X_train, y_train)



lasso_reg1 = rnd_search.best_estimator_

print(rnd_search.best_params_)



y_train_pred = lasso_reg1.predict(X_train)

print('-' * 10 + 'Lasso' + '-' * 10)

print('R square Accuracy for train: ', r2_score(y_train, y_train_pred))

print('Mean Absolute Error for train: ', mean_absolute_error(y_train, y_train_pred))

print('Mean Squared Error for train: ', mean_squared_error(y_train, y_train_pred))



y_pred = lasso_reg1.predict(X_test)

print('-' * 10 + 'Lasso' + '-' * 10)

print('R square Accuracy: ', r2_score(y_test, y_pred))

print('Mean Absolute Error: ', mean_absolute_error(y_test, y_pred))

print('Mean Squared Error: ', mean_squared_error(y_test, y_pred))
from sklearn.linear_model import ElasticNet
param_grid = [

        {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 2, 5, 10, 20]},

        {'l1_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}

    ]



elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)

grid_search = GridSearchCV(elastic_net, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)

grid_search.fit(X_train, y_train)

elastic_net = grid_search.best_estimator_

print(grid_search.best_params_)



y_train_pred = elastic_net.predict(X_train)

print('-' * 10 + 'ElasticNet' + '-' * 10)

print('R square Accuracy for train: ', r2_score(y_train, y_train_pred))

print('Mean Absolute Error for train: ', mean_absolute_error(y_train, y_train_pred))

print('Mean Squared Error for train: ', mean_squared_error(y_train, y_train_pred))



y_pred = elastic_net.predict(X_test)

print('-' * 10 + 'ElasticNet' + '-' * 10)

print('R square Accuracy: ', r2_score(y_test, y_pred))

print('Mean Absolute Error: ', mean_absolute_error(y_test, y_pred))

print('Mean Squared Error: ', mean_squared_error(y_test, y_pred))
elastic_net.coef_
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import expon, reciprocal



elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)



param_distribs = {

        'alpha': expon(scale=1.0),

        'l1_ratio': np.linspace(0, 1, 10)

    }



rnd_search = RandomizedSearchCV(elastic_net, param_distributions=param_distribs, cv=kfolds, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)

rnd_search.fit(X_train, y_train)



elastic_net1 = rnd_search.best_estimator_

print(rnd_search.best_params_)



y_train_pred = elastic_net1.predict(X_train)

print('-' * 10 + 'ElasticNet' + '-' * 10)

print('R square Accuracy for train: ', r2_score(y_train, y_train_pred))

print('Mean Absolute Error for train: ', mean_absolute_error(y_train, y_train_pred))

print('Mean Squared Error for train: ', mean_squared_error(y_train, y_train_pred))



y_pred = elastic_net1.predict(X_test)

print('-' * 10 + 'ElasticNet' + '-' * 10)

print('R square Accuracy: ', r2_score(y_test, y_pred))

print('Mean Absolute Error: ', mean_absolute_error(y_test, y_pred))

print('Mean Squared Error: ', mean_squared_error(y_test, y_pred))
#from mlxtend.regressor import StackingRegressor

from sklearn.ensemble import StackingRegressor
type(y_train)
type(X_train)
stregr = StackingRegressor(estimators=[('gb_reg', gb_reg1), 

                                       ('svr_reg1', svr_reg1), 

                                       ('lgb_reg1', lgb_reg1), 

                                       ('xgb_reg', xgb_reg1)], final_estimator=elastic_net1)

stregr.fit(X_train, y_train)



y_train_pred = stregr.predict(X_train)



print('-' * 10 + 'Stacking' + '-' * 10)

print('R square Accuracy for train: ', r2_score(y_train, y_train_pred))

print('Mean Absolute Error for train: ', mean_absolute_error(y_train, y_train_pred))

print('Mean Squared Error for train: ', mean_squared_error(y_train, y_train_pred))



y_pred = stregr.predict(X_test)

print('-' * 10 + 'Stacking' + '-' * 10)

print('R square Accuracy: ', r2_score(y_test, y_pred))

print('Mean Absolute Error: ', mean_absolute_error(y_test, y_pred))

print('Mean Squared Error: ', mean_squared_error(y_test, y_pred))
from sklearn.ensemble import VotingRegressor
named_estimators = [

    ("xgb_reg", xgb_reg),

    ("gb_reg1", gb_reg1),

    ("svr_reg1", svr_reg1),

    ("elastic_net1", elastic_net1),

]
voting_reg = VotingRegressor(named_estimators)
voting_reg.fit(X_train, y_train)
y_train_pred = voting_reg.predict(X_train)



print('-' * 10 + 'Voting' + '-' * 10)

print('R square Accuracy for train: ', r2_score(y_train, y_train_pred))

print('Mean Absolute Error for train: ', mean_absolute_error(y_train, y_train_pred))

print('Mean Squared Error for train: ', mean_squared_error(y_train, y_train_pred))



y_pred = voting_reg.predict(X_test)

print('-' * 10 + 'Voting' + '-' * 10)

print('R square Accuracy: ', r2_score(y_test, y_pred))

print('Mean Absolute Error: ', mean_absolute_error(y_test, y_pred))

print('Mean Squared Error: ', mean_squared_error(y_test, y_pred))
test_pred = stregr.predict(test)
test_pred
pd.DataFrame(test_pred).to_csv('submission_new.csv', index=False)