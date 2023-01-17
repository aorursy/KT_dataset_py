import numpy as np

import pandas as pd

import xgboost as xgb

from xgboost import plot_importance, plot_tree

from sklearn.metrics import mean_squared_error, mean_absolute_error
train = pd.read_csv("../input/covid19-global-forecasting-week-1/train.csv")

test = pd.read_csv("../input/covid19-global-forecasting-week-1/test.csv")

submission = pd.read_csv("../input/covid19-global-forecasting-week-1/submission.csv")
X_train = train.drop(["Fatalities", "ConfirmedCases"], axis=1)
X_train = X_train.drop(["Id"], axis=1)
X_train['Date']= pd.to_datetime(X_train['Date']) 
X_train = X_train.set_index(['Date'])
def create_time_features(df):

    """

    Creates time series features from datetime index

    """

    df['date'] = df.index

    df['hour'] = df['date'].dt.hour

    df['dayofweek'] = df['date'].dt.dayofweek

    df['quarter'] = df['date'].dt.quarter

    df['month'] = df['date'].dt.month

    df['year'] = df['date'].dt.year

    df['dayofyear'] = df['date'].dt.dayofyear

    df['dayofmonth'] = df['date'].dt.day

    df['weekofyear'] = df['date'].dt.weekofyear

    

    X = df[['hour','dayofweek','quarter','month','year',

           'dayofyear','dayofmonth','weekofyear']]

    return X
X_train_time = create_time_features(X_train)
X_train.drop("date", axis=1, inplace=True)
X_train = pd.concat([X_train,pd.get_dummies(X_train['Province/State'], prefix='ps')],axis=1)

X_train.drop(['Province/State'],axis=1, inplace=True)
X_train = pd.concat([X_train,pd.get_dummies(X_train['Country/Region'], prefix='cr')],axis=1)

X_train.drop(['Country/Region'],axis=1, inplace=True)
y_train = train["Fatalities"]
reg = xgb.XGBRegressor(n_estimators=1000)
reg.fit(X_train, y_train, verbose=True)
X_test = test.drop(["ForecastId"], axis=1)

X_test['Date']= pd.to_datetime(X_test['Date']) 

X_test = X_test.set_index(['Date'])

X_test_time = create_time_features(X_test)

X_test.drop("date", axis=1, inplace=True)



X_test = pd.concat([X_test,pd.get_dummies(X_test['Province/State'], prefix='ps')],axis=1)

X_test.drop(['Province/State'],axis=1, inplace=True)



X_test = pd.concat([X_test,pd.get_dummies(X_test['Country/Region'], prefix='cr')],axis=1)

X_test.drop(['Country/Region'],axis=1, inplace=True)
X_test['Fatalities'] = reg.predict(X_test)
X_test