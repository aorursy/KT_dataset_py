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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings('ignore',category=RuntimeWarning)

df_bike = pd.read_csv('../input/train.csv')

df_bike.tail()
df_bike.shape
df_bike.info()
df_bike['datetime'] = df_bike.datetime.apply(pd.to_datetime)



df_bike['year'] = df_bike.datetime.apply(lambda x : x.year)

df_bike['month'] = df_bike.datetime.apply(lambda x : x.month)

df_bike['day'] = df_bike.datetime.apply(lambda x : x.day)

df_bike['hour'] = df_bike.datetime.apply(lambda x : x.hour)

df_bike.tail(3)
drop_columns = ['datetime','casual','registered']

df_bike.drop(drop_columns, axis=1, inplace=True)

df_bike.tail()
from sklearn.metrics import mean_squared_error, mean_absolute_error



def rmsle(y, pred):

    log_y = np.log1p(y)

    log_pred = np.log1p(pred)

    squared_error = (log_y - log_pred)**2

    rmsle = np.sqrt(np.mean(squared_error))

    return rmsle



def rmse(y, pred):

    return np.sqrt(mean_squared_error(y,pred))



def evaluate_regr(y, pred):

    rmsle_val = rmsle(y, pred)

    rmse_val = rmse(y, pred)

    mse_val = mean_absolute_error(y, pred)

    print('RMSLE: {0:.3f}, RMSE: {1:.3f}, MSE: {2:.3f}'.format(rmsle_val, rmse_val, mse_val))
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LinearRegression, Ridge, Lasso



y_target = df_bike['count']

X_features = df_bike.drop(['count'], axis=1, inplace=False)
y_target.shape
X_features.shape
X_train, X_test, y_train, y_test = train_test_split(

    X_features, y_target, test_size=0.3, random_state=42)



lr_reg = LinearRegression()

lr_reg.fit(X_train, y_train)

pred = lr_reg.predict(X_test)



evaluate_regr(y_test, pred)
y_target.hist()
y_log_transform = np.log1p(y_target)

y_log_transform.hist()
y_target_log = np.log1p(y_target)



X_train, X_test, y_train, y_test = train_test_split(

    X_features, y_target_log, test_size=0.3, random_state=42)



lr_reg = LinearRegression()

lr_reg.fit(X_train, y_train)

pred = lr_reg.predict(X_test)
y_test_exp = np.expm1(y_test)

pred_exp = np.expm1(pred)

evaluate_regr(y_test_exp, pred_exp)
coef = pd.Series(lr_reg.coef_, index=X_features.columns)

coef_sort = coef.sort_values(ascending=False)

sns.barplot(x=coef_sort.values, y=coef_sort.index)
X_features_ohe = pd.get_dummies(

    X_features, columns=[

        'year','month','hour','holiday',

        'workingday','season','weather'])
X_train, X_test, y_train, y_test = train_test_split(

    X_features_ohe, y_target_log, test_size=0.3, random_state=42)
def get_model_predict(

    model, X_train, X_test, y_train, y_test, is_expm1=False):

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    if is_expm1:

        y_test = np.expm1(y_test)

        pred = np.expm1(pred)

    print('###', model.__class__.__name__, '###')

    evaluate_regr(y_test, pred)

    
lr_reg = LinearRegression()

ridge_reg = Ridge(alpha=10)

lasso_reg = Lasso(alpha=0.01)



for model in [lr_reg, ridge_reg, lasso_reg]:

    get_model_predict(model, X_train, X_test, y_train, y_test, is_expm1=True)

coef = pd.Series(lr_reg.coef_, index=X_features_ohe.columns)

coef_sort = coef.sort_values(ascending=False)[:10]

sns.barplot(x=coef_sort.values, y=coef_sort.index)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from lightgbm import LGBMRegressor



rf_reg = RandomForestRegressor(n_estimators=500)

gbm_reg = GradientBoostingRegressor(n_estimators=500)

lgbm_reg = LGBMRegressor(n_estimators=500)



for model in [rf_reg, gbm_reg, lgbm_reg]:

    get_model_predict(model, X_train, X_test, y_train, y_test, is_expm1=True)