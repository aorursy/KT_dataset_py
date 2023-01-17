# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
housing = pd.read_csv('../input/train.csv')
housing.head()
housing.info()
housing.describe()
%matplotlib inline
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()
corr_matrix = housing.corr()
corr_matrix['SalePrice'].sort_values(ascending=False)
from pandas.plotting import scatter_matrix
attributes = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF']
scatter_matrix(housing[attributes], figsize=(12, 12))
quantitative = [f for f in housing.columns if housing.dtypes[f] != 'object']
quantitative.remove('SalePrice')
quantitative.remove('Id')
qualitative = [f for f in housing.columns if housing.dtypes[f] == 'object']
# check which columns has missing values
missing_columns = [column for column in quantitative if housing[column].isna().sum() > 0]
missing_columns
housing[missing_columns].hist()
X = housing[quantitative].values
X
np.argwhere(np.isnan(X))[:10]
from sklearn.preprocessing import Imputer

imputer = Imputer()

imputer.fit(X)

imputer_X = imputer.transform(X)

np.argwhere(np.isnan(imputer_X))
X = imputer_X
y = housing[['SalePrice']]
y[0:10]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn import linear_model
ridge = linear_model.Ridge()
ridge.fit(X_train, y_train)
def error(actual, predicted):
    actual = np.log(actual)
    predicted = np.log(predicted)
    return np.sqrt(np.sum(np.square(actual-predicted))/len(actual))
ridge.score(X_train, y_train)
error(y_train, ridge.predict(X_train))
error(y_test, ridge.predict(X_test))
ridge.score(X_test, y_test)
lasso = linear_model.Lasso()
lasso.fit(X_train, y_train)
lasso.score(X_train, y_train)
lasso.score(X_test, y_test)
import xgboost as xgb
xgb_reg = xgb.XGBRegressor(n_jobs=2)
xgb_reg.fit(X_train, y_train)
xgb_reg.score(X_train, y_train)
xgb_reg.score(X_test, y_test)
test = pd.read_csv('../input/test.csv')
test_ids = test['Id']
test.drop(columns=['Id'])
xgb_submit1 = xgb.XGBRegressor(n_jobs=2)
xgb_submit1.fit(X, y)

test_X = test[quantitative].values
#test_X = imputer.fit(test_X)
predicted_prices = xgb_submit1.predict(test_X)
predicted_prices
my_submission = pd.DataFrame({'Id': test_ids, 'SalePrice': predicted_prices})
my_submission.to_csv('submission1.csv', index=False)
from category_encoders import OneHotEncoder

category_data = housing.drop(columns=quantitative + ['Id', 'SalePrice'])
encoder = OneHotEncoder().fit(category_data, y)
encoder.category_mapping
encoded_X = encoder.transform(category_data)
encoded_X.head()
encoded_X.isna().sum().sum()
imputer_X.shape
encoded_X.shape
X = np.append(imputer_X, encoded_X, axis=1)
X.shape
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X, y, test_size=0.2)
xgb_cat = xgb.XGBRegressor(n_jobs=2)
xgb_cat.fit(X_train_cat, y_train_cat)
xgb_cat.score(X_train_cat, y_train_cat)
xgb_cat.score(X_test_cat, y_test_cat)
import xgboost as xgb
xgb.plot_importance(xgb_cat, max_num_features=20)
xgb.plot_tree(xgb_cat)
xgb_submit2 = xgb.XGBRegressor(n_jobs=2)
xgb_submit2.fit(X, y)
encoded_test = encoder.transform(test[qualitative])
trasnformed_test = np.append(test[quantitative], encoded_test, axis=1)
predicted_prices2 = xgb_submit2.predict(trasnformed_test)
submission2 = pd.DataFrame({'Id': test_ids, 'SalePrice': predicted_prices2})
submission2.to_csv('submission2.csv', index=False)
from sklearn.model_selection import cross_val_score
xgb_cat = xgb.XGBRegressor(n_jobs=2)
scores = cross_val_score(xgb_cat, X, y, cv=5, n_jobs=1, verbose=2)
print('Scores:', scores)
print('Mean:', scores.mean())
print('Standard deviation:', scores.std())
# https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

from sklearn.model_selection import GridSearchCV

param_grid={ 
    'max_depth': [3],
    'learning_rate': [0.1],
    'n_estimators': [100],
    'gamma': [0],
    'min_child_weight': [1],
    'max_delta_step': [0],
    'subsample': [1],
    'colsample_by_tree': [1],
    'colsample_bylevel': [1]
}
xgb_search0 = xgb.XGBRegressor(
    n_estimators=1000,
    n_jobs=2
)
xgb_search0.fit(
    X_train_cat, y_train_cat,
    early_stopping_rounds=10,
    eval_set= [[X_test_cat, y_test_cat]]
)
xgb_search0.score(X_test_cat, y_test_cat)
param1= {
    'max_depth': range(3,6),
    'min_child_weight': range(1,6,2)
}
xgb_search1 = xgb.XGBRegressor(n_estimators=147, n_jobs=2)
grid_search1 = GridSearchCV(
    xgb_search1, param1,
    cv=5,
    verbose=1,
#     scoring='neg_mean_squared_error'
)
grid_search1.fit(X, y)
grid_search1.best_score_
grid_search1.best_params_
param2 = {
    'gamma': [i/10.0 for i in range(0,5)]
}
xgb_search2 = xgb.XGBRegressor(n_estimators=147, n_jobs=2)
grid_search2 = GridSearchCV(
    xgb_search2, param2, 
    cv=5,
    verbose=1,
#     scoring='neg_mean_squared_error'
)
grid_search2.fit(X, y)
grid_search2.best_score_
grid_search2.best_params_
param3= {
    'subsample': [i/10.0 for i in range(6,10)],
    'colsample_bytree': [i/10.0 for i in range(6,10)]
}
xgb_search3 = xgb.XGBRegressor(n_estimators=147, n_jobs=2)
grid_search3 = GridSearchCV(
    xgb_search3, param3, 
    cv=5,
    verbose=1,
#     scoring='neg_mean_squared_error'
)
grid_search3.fit(X, y)
grid_search3.best_score_
grid_search3.best_params_
xgb_search4 = xgb.XGBRegressor(
    learning_rate=0.05,
    n_estimators=1000,
    subsample= 0.7,
    colsample_bytree=0.8,
    n_jobs=2
)
xgb_search4.fit(
    X_train_cat, y_train_cat,
    early_stopping_rounds=10,
    eval_set= [[X_test_cat, y_test_cat]]
)
xgb_search4.score(X_test_cat, y_test_cat)
param3= {
    'subsample': [i/10.0 for i in range(6,10)],
    'colsample_bytree': [i/10.0 for i in range(6,10)]
}
xgb_search5 = xgb.XGBRegressor(
    learnin_rate=0.05,
    n_estimators=188, n_jobs=2)
grid_search5 = GridSearchCV(
    xgb_search5, param3, 
    cv=5,
    verbose=1,
#     scoring='neg_mean_squared_error'
)
grid_search5.fit(X, y)
grid_search5.best_score_
grid_search5.best_params_
predicted_prices3 = grid_search5.best_estimator_.predict(trasnformed_test)
submission3 = pd.DataFrame({'Id': test_ids, 'SalePrice': predicted_prices3})
submission3.to_csv('submission3.csv', index=False)
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
scaler.fit(X)
scaled_X = scaler.transform(X)
xgb_best = xgb.XGBRegressor(
    learning_rate=0.05,
    n_estimators=188,
    subsample= 0.7,
    colsample_bytree=0.8,
    n_jobs=2
)
scores_encoded = cross_val_score(xgb_best, X, y, cv=5, verbose=3)
scores_scaled = cross_val_score(xgb_best, scaled_X, y, cv=5, verbose=3)
print('Scores:', scores_encoded)
print('Mean:', scores_encoded.mean())
print('Standard deviation:', scores_encoded.std())
print('Scores:', scores_scaled)
print('Mean:', scores_scaled.mean())
print('Standard deviation:', scores_scaled.std())
scaled_X_train, scaled_X_test, scaled_y_train, scaled_y_test = train_test_split(scaled_X, y, test_size=0.2)
xgb_best.fit(scaled_X_train, scaled_y_train)
xgb_predictions = xgb_best.predict(scaled_X_test)
from sklearn.metrics import r2_score
r2_score(scaled_y_test, xgb_predictions)
# lasso = linear_model.LassoCV(cv=5, verbose=True)
lasso = linear_model.Lasso()
lasso.fit(scaled_X_train, scaled_y_train)
lasso_predictions = lasso.predict(scaled_X_test)
r2_score(scaled_y_test, lasso_predictions)
stacked = lasso_predictions * 0.5 + xgb_predictions * 0.5
r2_score(scaled_y_test,stacked)
stacked_predictions = [lasso_predictions * (i / 10.0) + xgb_predictions * ((10 - i)/ 10)for i in range(10)]
stacked_scores = [r2_score(scaled_y_test, stacked) for stacked in stacked_predictions]
stacked_scores
best_i = np.argmax(stacked_scores)
best_i, stacked_scores[best_i]