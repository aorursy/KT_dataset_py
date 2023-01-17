import numpy as np

import pandas as pd

import xgboost as xgb

from xgboost import plot_importance, plot_tree

from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.ensemble import RandomForestRegressor
train = pd.read_csv("../input/covid19-global-forecasting-week-1/train.csv")

test = pd.read_csv("../input/covid19-global-forecasting-week-1/test.csv")
X_train = train.drop(["Fatalities", "ConfirmedCases"], axis=1)
countries = X_train["Country/Region"]
X_train = X_train.drop(["Id"], axis=1)

X_test = test.drop(["ForecastId"], axis=1)
X_train['Date']= pd.to_datetime(X_train['Date']) 

X_test['Date']= pd.to_datetime(X_test['Date']) 
X_train = X_train.set_index(['Date'])

X_test = X_test.set_index(['Date'])
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
create_time_features(X_train)

create_time_features(X_test)
X_train.drop("date", axis=1, inplace=True)

X_test.drop("date", axis=1, inplace=True)
X_train = pd.concat([X_train,pd.get_dummies(X_train['Province/State'], prefix='ps')],axis=1)

X_train.drop(['Province/State'],axis=1, inplace=True)

X_test = pd.concat([X_test,pd.get_dummies(X_test['Province/State'], prefix='ps')],axis=1)

X_test.drop(['Province/State'],axis=1, inplace=True)
X_train = pd.concat([X_train,pd.get_dummies(X_train['Country/Region'], prefix='cr')],axis=1)

X_train.drop(['Country/Region'],axis=1, inplace=True)

X_test = pd.concat([X_test,pd.get_dummies(X_test['Country/Region'], prefix='cr')],axis=1)

X_test.drop(['Country/Region'],axis=1, inplace=True)
y_train = train["Fatalities"]
# reg = xgb.XGBRegressor(n_estimators=1000,max_depth=10,silent=0,nthread=6,verbosity=2,num_parallel_tree=10,n_jobs=1000)
# reg.fit(X_train, y_train, verbose=True)
# plot = plot_importance(reg, height=0.9)
# y_train = train["ConfirmedCases"]

# confirmed_reg = xgb.XGBRegressor(n_estimators=1000,max_depth=10,silent=0,nthread=6,verbosity=2,num_parallel_tree=5,n_jobs=100)

# confirmed_reg.fit(X_train, y_train, verbose=True)

# preds = confirmed_reg.predict(X_test)

# preds = np.array(preds)

# preds[preds < 0] = 0

# preds = np.round(preds, 0)
# preds = np.array(preds)
submissionOrig = pd.read_csv("../input/covid19-global-forecasting-week-1/submission.csv")
y_train = train["ConfirmedCases"]

confirmed_reg = RandomForestRegressor(max_depth=100,n_jobs=-1,n_estimators=100)

confirmed_reg.fit(X_train, y_train)

preds = confirmed_reg.predict(X_test)

preds = np.array(preds)

preds[preds < 0] = 0

preds = np.round(preds, 0)

preds = np.array(preds)

submissionOrig["ConfirmedCases"]=pd.Series(preds)
y_train = train["Fatalities"]

confirmed_reg = RandomForestRegressor(max_depth=100,n_jobs=-1,n_estimators=100)

confirmed_reg.fit(X_train, y_train)

preds = confirmed_reg.predict(X_test)

preds = np.array(preds)

preds[preds < 0] = 0

preds = np.round(preds, 0)

submissionOrig["Fatalities"]=pd.Series(preds)



submissionOrig.to_csv('submission.csv',index=False)