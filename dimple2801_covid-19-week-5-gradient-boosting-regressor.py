# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



import statsmodels.api as sm

from statsmodels.tsa.arima_model import ARIMA

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score

from xgboost import XGBRegressor 

from sklearn.ensemble import GradientBoostingRegressor



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
parse_date = lambda val : pd.datetime.strptime(val, '%y-%m-%d')

df_train = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv', parse_dates = ['Date'])

df_test = pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv', parse_dates = ['Date'])
df_train['week_day'] = df_train['Date'].apply(lambda val: val.day_name())

df_train['month'] = df_train['Date'].apply(lambda val: val.month)



df_test['week_day'] = df_test['Date'].apply(lambda val: val.day_name())

df_test['month'] = df_test['Date'].apply(lambda val: val.month)
one_hot = pd.get_dummies(df_train['week_day'])



# df_train = df_train.drop('week_day',axis = 1)

df_train = df_train.join(one_hot)



df_train.tail()
one_hot_test = pd.get_dummies(df_test['week_day'])



df_test = df_test.drop('week_day',axis = 1)

df_test = df_test.join(one_hot_test)

df_test.head()
df_train['is_weekend'] = df_train['Saturday'] + df_train['Sunday']

df_test['is_weekend'] = df_test['Saturday'] + df_test['Sunday']
def fill_na_county_value(row):

    if pd.isnull(row['Province_State']) and pd.isnull(row['County']):

        val = row['Country_Region']

    elif pd.isnull(row['County']):

        val = str(row['Country_Region']) + "_" + str(row['Province_State'])

    else:

        val = str(row['Country_Region']) + "_" + str(row['Province_State']) + "_" + row['County']

    return val



df_train['County'] = df_train.apply(fill_na_county_value, axis=1)

df_test['County'] = df_test.apply(fill_na_county_value, axis=1)
df_submission = pd.DataFrame(columns=['ForecastId_Quantile', 'TargetValue'])



for name in df_train.County.unique():

    df_train_county = "df_train_{0}.format(name)"

    df_train_county = df_train[df_train['County']==name]

    

    df_test_county = "df_test_{0}.format(name)"

    df_test_county = df_test[df_test['County']==name]

    

    df_train_county_cases = df_train_county[df_train_county['Target'] == 'ConfirmedCases']

    df_test_county_cases = df_test_county[df_test_county['Target'] == 'ConfirmedCases']

    test_county_cases_index = df_test_county_cases.ForecastId

    

    df_train_county_deaths = df_train_county[df_train_county['Target'] == 'Fatalities']

    df_test_county_deaths = df_test_county[df_test_county['Target'] == 'Fatalities']

    test_county_deaths_index = df_test_county_deaths.ForecastId

    



    X_train_cases = df_train_county_cases[['month', 'Sunday', 'Monday', 'Tuesday', 'Wednesday',

                                           'Thursday', 'Friday', 'Saturday', 'is_weekend']]

    y_train_cases = df_train_county_cases[['TargetValue']]

    

    X_test_cases = df_test_county_cases[['month', 'Sunday', 'Monday', 'Tuesday', 'Wednesday',

                                           'Thursday', 'Friday', 'Saturday', 'is_weekend']]

    y_test_cases = df_test_county_cases[['ForecastId']]



    gbm = GradientBoostingRegressor(loss='quantile', alpha=0.95,

                                    n_estimators=250, max_depth=3,

                                    learning_rate=.1, min_samples_leaf=9,

                                    min_samples_split=9)

    

    gbm.fit(X_train_cases, y_train_cases)



    # Predicting upper threshold

    y_upper_cases = gbm.predict(X_test_cases)

    y_test_cases['ForecastId_Quantile'] = test_county_cases_index.apply(lambda val: str(val) + '_0.95')

    y_test_cases['TargetValue'] = y_upper_cases

    df_submission = df_submission.append(y_test_cases)

    

    # Predicting lower threshold

    gbm.set_params(alpha=0.05)

    gbm.fit(X_train_cases, y_train_cases)

    y_lower_cases = gbm.predict(X_test_cases)

    y_test_cases['ForecastId_Quantile'] = test_county_cases_index.apply(lambda val: str(val) + '_0.05')

    y_test_cases['TargetValue'] = y_lower_cases

    df_submission = df_submission.append(y_test_cases)

    

    # Predicting actual cases

    gbm.set_params(loss='ls')

    gbm.fit(X_train_cases, y_train_cases)



    # Make the prediction on the meshed x-axis

    y_pred_cases = gbm.predict(X_test_cases)

    y_test_cases['ForecastId_Quantile'] = test_county_cases_index.apply(lambda val: str(val) + '_0.5')

    y_test_cases['TargetValue'] = y_pred_cases

    df_submission = df_submission.append(y_test_cases)

    

    

    

    # predicting deaths

    

    X_train_deaths = df_train_county_deaths[['month', 'Sunday', 'Monday', 'Tuesday', 'Wednesday',

                                           'Thursday', 'Friday', 'Saturday', 'is_weekend']]

    y_train_deaths = df_train_county_deaths[['TargetValue']]

    

    X_test_deaths = df_test_county_deaths[['month', 'Sunday', 'Monday', 'Tuesday', 'Wednesday',

                                           'Thursday', 'Friday', 'Saturday', 'is_weekend']]

    y_test_deaths = df_test_county_deaths[['ForecastId']]

    

    gbm = GradientBoostingRegressor(loss='quantile', alpha=0.95,

                                    n_estimators=250, max_depth=3,

                                    learning_rate=.1, min_samples_leaf=9,

                                    min_samples_split=9)

    

    gbm.fit(X_train_deaths, y_train_deaths)



    # Predicting upper threshold

    y_upper_deaths = gbm.predict(X_test_deaths)

    y_test_deaths['ForecastId_Quantile'] = test_county_deaths_index.apply(lambda val: str(val) + '_0.95')

    y_test_deaths['TargetValue'] = y_upper_deaths

    df_submission = df_submission.append(y_test_deaths)

    

    # Predicting lower threshold

    gbm.set_params(alpha=0.05)

    gbm.fit(X_train_deaths, y_train_deaths)

    y_lower_deaths = gbm.predict(X_test_deaths)

    y_test_deaths['ForecastId_Quantile'] = test_county_deaths_index.apply(lambda val: str(val) + '_0.05')

    y_test_deaths['TargetValue'] = y_lower_deaths

    df_submission = df_submission.append(y_test_deaths)



    # Predicting actual deaths

    gbm.set_params(loss='ls')

    gbm.fit(X_train_deaths, y_train_deaths)



    # Make the prediction on the meshed x-axis

    y_pred_deaths = gbm.predict(X_test_deaths)

    y_test_deaths['ForecastId_Quantile'] = test_county_deaths_index.apply(lambda val: str(val) + '_0.5')

    y_test_deaths['TargetValue'] = y_pred_deaths

    df_submission = df_submission.append(y_test_deaths)



    
df_submission = df_submission.drop('ForecastId', axis=1)

df_submission.head()
df_submission.to_csv("submission.csv", index=False)