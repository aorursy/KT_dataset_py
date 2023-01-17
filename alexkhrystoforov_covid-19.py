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
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

sub = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')
train['Date'] = pd.to_datetime(train['Date'], infer_datetime_format=True)

test['Date'] = pd.to_datetime(test['Date'], infer_datetime_format=True)
train.loc[:, 'Date'] = train.Date.dt.strftime('%y%m%d')

train.loc[:, 'Date'] = train['Date'].astype(int)



test.loc[:, 'Date'] = test.Date.dt.strftime('%y%m%d')

test.loc[:, 'Date'] = test['Date'].astype(int)
train['Province_State'].fillna('nan', inplace=True)

test['Province_State'].fillna('nan', inplace=True)
from sklearn.preprocessing import LabelEncoder
#get list of categorical variables

s = (train.dtypes == 'object')

object_cols = list(s[s].index)
label_encoder1 = LabelEncoder()

label_encoder2 = LabelEncoder()



train['Province_State'] = label_encoder1.fit_transform(train['Province_State'])

test['Province_State'] = label_encoder1.transform(test['Province_State'])



train['Country_Region'] = label_encoder2.fit_transform(train['Country_Region'])

test['Country_Region'] = label_encoder2.transform(test['Country_Region'])

from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.metrics import mean_squared_log_error
X_train = train[['Province_State','Country_Region','Date']]

X_train_with_confirmed = train[['Province_State','Country_Region','Date','ConfirmedCases']]

y_train_full = train[['ConfirmedCases', 'Fatalities']]

y_train_confirmed = train['ConfirmedCases']

y_train_fatal = train['Fatalities']
sub1 = pd.DataFrame()

sub1['ForecastID'] = test['ForecastId']
del test['ForecastId']
# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train_confirmed, test_size=0.2, random_state=123)
xgb = XGBRegressor(

    n_estimators = 500,

    max_depth = 20,

    learning_rate = 0.1, 

)
xgb.fit(X_train, y_train_confirmed)
test_y_conf = xgb.predict(test)



test_y_conf[test_y_conf < 0] = 0



# print("error",'\t',mean_squared_log_error(y_test, preds))
test_y_conf = pd.DataFrame(test_y_conf)
test_y_conf.rename(columns={0: 'ConfirmedCases'}, inplace=True)
sub1 = pd.concat([sub1, test_y_conf], axis=1)
test = pd.concat([test, test_y_conf], axis=1)
test
# X_train, X_test, y_train, y_test = train_test_split(X_train_with_confirmed, y_train_fatal, test_size=0.2, random_state=123)
xgb1 = XGBRegressor(

    n_estimators = 500,

    max_depth = 20,

    learning_rate = 0.1, 

)
xgb1.fit(X_train_with_confirmed, y_train_fatal)
test_y_fatal = xgb1.predict(test)



# test_y_fatal[test_y_conf < 0] = 0
test_y_fatal = pd.DataFrame(test_y_fatal)



test_y_fatal.rename(columns={0: 'Fatalities'}, inplace=True)

test_y_fatal[test_y_fatal < 0] = 0
sub1 = pd.concat([sub1, test_y_fatal], axis=1)
sub1.head()
sub1.to_csv("submission.csv" , index = False)
# def algorithm_pipeline(X_train_data, X_test_data, y_train_data, y_test_data, 

#                        model, param_grid, cv=10, scoring_fit='neg_mean_squared_error',

#                        do_probabilities = False):

#     gs = GridSearchCV(

#         estimator=model,

#         param_grid=param_grid, 

#         cv=cv, 

#         n_jobs=-1, 

#         scoring=scoring_fit,

#         verbose=2

#     )

#     fitted_model = gs.fit(X_train_data, y_train_data)

    

#     if do_probabilities:

#         pred = fitted_model.predict_proba(X_test_data)

#     else:

#         pred = fitted_model.predict(X_test_data)

    

#     return fitted_model, pred
# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train_confirmed, test_size=0.2, random_state=42)
# xgb = XGBRegressor()



# param_grid = {

#     'n_estimators': [400, 700, 1000],

#     'colsample_bytree': [0.7, 0.8],

#     'max_depth': [15,20,25],

#     'reg_alpha': [1.1, 1.2, 1.3],

#     'reg_lambda': [1.1, 1.2, 1.3],

#     'subsample': [0.7, 0.8, 0.9]

# }



# model, pred = algorithm_pipeline(X_train, X_test, y_train, y_test, xgb, 

#                                  param_grid, cv=5)



# # Root Mean Squared Error

# print(np.sqrt(-model.best_score_))

# print(model.best_params_)