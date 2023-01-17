import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
# import tensorflow as tf
import math
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
import datetime as dt
from sklearn.preprocessing import OrdinalEncoder
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_log_error
train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")
test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")
train['Date'] = pd.to_datetime(train['Date'], format = '%Y-%m-%d')
test['Date'] = pd.to_datetime(test['Date'], format = '%Y-%m-%d')
train[['ConfirmedCases', 'Fatalities']].describe()
def create_features(df):
    df['day'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['quarter'] = df['Date'].dt.quarter
    df['weekofyear'] = df['Date'].dt.weekofyear
    return df
def train_dev_split(df, days):
    date = df['Date'].max() - dt.timedelta(days=days)
    return df[df['Date'] <= date], df[df['Date'] > date]
def categoricalToInteger(df):
    df.Province_State.fillna('None', inplace=True)
    oe = OrdinalEncoder()
    df[['Province_State','Country_Region']] = oe.fit_transform(df.loc[:,['Province_State','Country_Region']])
    return df
df_train = categoricalToInteger(train)
df_train.info()
df_train = create_features(df_train)
df_train, df_dev = train_dev_split(df_train,0)
columns = ['day','month','dayofweek','dayofyear','quarter','weekofyear','Province_State', 'Country_Region','ConfirmedCases','Fatalities']
df_train = df_train[columns]
df_dev = df_dev[columns]
df_test = categoricalToInteger(test)
df_test = create_features(test)
columns = ['day','month','dayofweek','dayofyear','quarter','weekofyear','Province_State', 'Country_Region']
def XGB():
    model = XGBRegressor(n_estimators=1300)
    return model
submission = []
for country in df_train.Country_Region.unique():
    df_train1 = df_train[df_train["Country_Region"]==country]
    for state in df_train1.Province_State.unique():
        df_train2 = df_train1[df_train1["Province_State"]==state]
        train = df_train2.values
        X_train, y_train = train[:,:-2], train[:,-2:]
        # lin_model1 = RidgeCV().fit(X_train, y_train[:,0])
        # lin_model1 = RidgeCV().fit(X_train, y_train[:,1])
        # lin_model1 = LinearRegression().fit(X_train, y_train[:,0])
        # lin_model2= LinearRegression().fit(X_train, y_train[:,1])
        model1 = XGBRegressor(n_estimators=1100)   #model 1 predicts Confirmed Cases
        model1.fit(X_train, y_train[:,0])
        model2 = XGBRegressor(n_estimators=1100)  #model 2 predicts Fatalities
        model2.fit(X_train, y_train[:,1])
        df_test1 = df_test[(df_test["Country_Region"]==country) & (df_test["Province_State"] == state)]
        ForecastId = df_test1.ForecastId.values
        df_test2 = df_test1[columns]
        y_pred1 = np.round(model1.predict(df_test2.values),5)
        y_pred2 = np.round(model2.predict(df_test2.values),5)
        for i in range(len(y_pred1)):
            d = {'ForecastId':ForecastId[i], 'ConfirmedCases':y_pred1[i], 'Fatalities':y_pred2[i]}
            submission.append(d)
df_submit = pd.DataFrame(submission)

df_submit.to_csv(r'submission.csv', index=False)