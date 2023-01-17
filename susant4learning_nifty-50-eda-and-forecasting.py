# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/nifty50-stock-market-data/DRREDDY.csv")

df.head()
data = df[['Date','Open','High','Low','Close','Volume','VWAP']]
df.set_index("Date", drop=False, inplace=True)
data
df.VWAP.plot(figsize=(14, 7))
data.info()

df.reset_index(drop=True, inplace=True)

lag_features = ["High", "Low", "Volume", "Turnover", "Trades"]

window1 = 3

window2 = 7

window3 = 30



df_rolled_3d = df[lag_features].rolling(window=window1, min_periods=0)

df_rolled_7d = df[lag_features].rolling(window=window2, min_periods=0)

df_rolled_30d = df[lag_features].rolling(window=window3, min_periods=0)



df_mean_3d = df_rolled_3d.mean().shift(1).reset_index().astype(np.float32)

df_mean_7d = df_rolled_7d.mean().shift(1).reset_index().astype(np.float32)

df_mean_30d = df_rolled_30d.mean().shift(1).reset_index().astype(np.float32)



df_std_3d = df_rolled_3d.std().shift(1).reset_index().astype(np.float32)

df_std_7d = df_rolled_7d.std().shift(1).reset_index().astype(np.float32)

df_std_30d = df_rolled_30d.std().shift(1).reset_index().astype(np.float32)



for feature in lag_features:

    df[f"{feature}_mean_lag{window1}"] = df_mean_3d[feature]

    df[f"{feature}_mean_lag{window2}"] = df_mean_7d[feature]

    df[f"{feature}_mean_lag{window3}"] = df_mean_30d[feature]

    

    df[f"{feature}_std_lag{window1}"] = df_std_3d[feature]

    df[f"{feature}_std_lag{window2}"] = df_std_7d[feature]

    df[f"{feature}_std_lag{window3}"] = df_std_30d[feature]



df.fillna(df.mean(), inplace=True)



df.set_index("Date", drop=False, inplace=True)

df.head()
# Convert string to datetime64

data['Date'] = data['Date'].apply(pd.to_datetime)

data.set_index('Date',inplace=True)

data.head()

df.Date = pd.to_datetime(df.Date, format="%Y-%m-%d")

df["month"] = df.Date.dt.month

df["week"] = df.Date.dt.week

df["day"] = df.Date.dt.day

df["day_of_week"] = df.Date.dt.dayofweek

df.head()
df_train = df[df.Date < "2019"]

df_valid = df[df.Date >= "2019"]



exogenous_features = ["High_mean_lag3", "High_std_lag3", "Low_mean_lag3", "Low_std_lag3",

                      "Volume_mean_lag3", "Volume_std_lag3", "Turnover_mean_lag3",

                      "Turnover_std_lag3", "Trades_mean_lag3", "Trades_std_lag3",

                      "High_mean_lag7", "High_std_lag7", "Low_mean_lag7", "Low_std_lag7",

                      "Volume_mean_lag7", "Volume_std_lag7", "Turnover_mean_lag7",

                      "Turnover_std_lag7", "Trades_mean_lag7", "Trades_std_lag7",

                      "High_mean_lag30", "High_std_lag30", "Low_mean_lag30", "Low_std_lag30",

                      "Volume_mean_lag30", "Volume_std_lag30", "Turnover_mean_lag30",

                      "Turnover_std_lag30", "Trades_mean_lag30", "Trades_std_lag30",

                      "month", "week", "day", "day_of_week"]
!pip install pmdarima
import lightgbm as lgb

import numpy as np

import pandas as pd



from fbprophet import Prophet

from matplotlib import pyplot as plt

from pmdarima import auto_arima

from sklearn.metrics import mean_absolute_error, mean_squared_error



myfavouritenumber = 37

seed = myfavouritenumber

np.random.seed(seed)
model = auto_arima(df_train.VWAP, exogenous=df_train[exogenous_features], trace=True, error_action="ignore", suppress_warnings=True)

model.fit(df_train.VWAP, exogenous=df_train[exogenous_features])



forecast = model.predict(n_periods=len(df_valid), exogenous=df_valid[exogenous_features])

df_valid["Forecast_ARIMAX"] = forecast
df_valid[["VWAP", "Forecast_ARIMAX"]].plot(figsize=(14, 7))
print("RMSE of Auto ARIMAX:", np.sqrt(mean_squared_error(df_valid.VWAP, df_valid.Forecast_ARIMAX)))

print("\nMAE of Auto ARIMAX:", mean_absolute_error(df_valid.VWAP, df_valid.Forecast_ARIMAX))
model_fbp = Prophet()

for feature in exogenous_features:

    model_fbp.add_regressor(feature)



model_fbp.fit(df_train[["Date", "VWAP"] + exogenous_features].rename(columns={"Date": "ds", "VWAP": "y"}))



forecast = model_fbp.predict(df_valid[["Date", "VWAP"] + exogenous_features].rename(columns={"Date": "ds"}))

df_valid["Forecast_Prophet"] = forecast.yhat.values
model_fbp.plot_components(forecast)
df_valid[["VWAP", "Forecast_ARIMAX", "Forecast_Prophet"]].plot(figsize=(14, 7))
print("RMSE of Auto ARIMAX:", np.sqrt(mean_squared_error(df_valid.VWAP, df_valid.Forecast_ARIMAX)))

print("RMSE of Prophet:", np.sqrt(mean_squared_error(df_valid.VWAP, df_valid.Forecast_Prophet)))

print("\nMAE of Auto ARIMAX:", mean_absolute_error(df_valid.VWAP, df_valid.Forecast_ARIMAX))

print("MAE of Prophet:", mean_absolute_error(df_valid.VWAP, df_valid.Forecast_Prophet))
params = {"objective": "regression"}



dtrain = lgb.Dataset(df_train[exogenous_features], label=df_train.VWAP.values)

dvalid = lgb.Dataset(df_valid[exogenous_features])



model_lgb = lgb.train(params, train_set=dtrain)



forecast = model_lgb.predict(df_valid[exogenous_features])

df_valid["Forecast_LightGBM"] = forecast
df_valid[["VWAP", "Forecast_ARIMAX", "Forecast_Prophet", "Forecast_LightGBM"]].plot(figsize=(14, 7))
print("RMSE of Auto ARIMAX:", np.sqrt(mean_squared_error(df_valid.VWAP, df_valid.Forecast_ARIMAX)))

print("RMSE of Prophet:", np.sqrt(mean_squared_error(df_valid.VWAP, df_valid.Forecast_Prophet)))

print("RMSE of LightGBM:", np.sqrt(mean_squared_error(df_valid.VWAP, df_valid.Forecast_LightGBM)))

print("\nMAE of Auto ARIMAX:", mean_absolute_error(df_valid.VWAP, df_valid.Forecast_ARIMAX))

print("MAE of Prophet:", mean_absolute_error(df_valid.VWAP, df_valid.Forecast_Prophet))

print("MAE of LightGBM:", mean_absolute_error(df_valid.VWAP, df_valid.Forecast_LightGBM))
#Thanks Vopani for your Kernal, explored some nice Concepts.