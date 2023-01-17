import os

import pandas as pd

import numpy as np

from matplotlib import rcParams

import matplotlib.pyplot as plt

import statsmodels.api as sm

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults

from statsmodels.tsa.stattools import adfuller

from sklearn.metrics import mean_squared_error

os.getcwd()
os.chdir('..')

os.listdir()
os.chdir("input")

os.listdir()
#path = r"C:\Users\[..]\Desktop\pump-sensor-data"

#df = pd.read_csv(os.path.abspath(path + r"/sensor.csv"),index_col = "timestamp",parse_dates=["timestamp"])

df = pd.read_csv('sensor.csv',index_col = "timestamp",parse_dates=["timestamp"])

df.drop("Unnamed: 0",axis=1,inplace=True)

df.head()
df['machine_status'].unique()
df.info()
df.drop('sensor_15',axis=1,inplace=True)

df["machine_status"]=df.machine_status.astype("category")
df.machine_status.dtype
df.describe()
df["sensor_00"].plot(figsize=(10,6))

plt.xticks(color="white")

plt.yticks(color="white")

plt.title("Capteur 0 série temporelle", color="white")

plt.xlabel("Timestamp", color="white")
df[(df.index.month>=4) & (df.index.month<=6)]["sensor_00"].plot(figsize=(10,7))

plt.xticks(color="white")

plt.yticks(color="white")

plt.xlabel("Timestamp", color="white")

plt.title("Extract of the time series between April and June for sensor 00",color = "white")
df["machine_status"].cat.categories
df["machine_status_code"]=df["machine_status"].cat.codes
df["machine_status"].cat.codes.unique()
df["machine_status"].cat.codes.plot(figsize=(10,7))
df.loc[df["machine_status"]=="BROKEN",["machine_status","machine_status_code"]]
df.loc[df["machine_status"]=="RECOVERING",["machine_status","machine_status_code"]].head()
df.loc[df["machine_status"]=="NORMAL",["machine_status","machine_status_code"]].head()
df[(df.index.month>=4) & (df.index.month<=5)].plot(figsize=(15,120), subplots=True)
#On sélectionne une partie de notre dataset pour entrainer notre modèle supervisé

df_train = df[(df.index.month>=4) & (df.index.month<=5)]
df_train["sensor_00"].fillna(method='bfill',inplace=True)
df_train["sensor_00"].isna().sum()
df_train[["sensor_00","machine_status_code"]].plot(figsize=(10,6),subplots=True)

plt.title("Capteur 00 et status de fonctionnement de la machine", color="white")

plt.xticks(color="white",rotation=0)

plt.yticks(color="white")

plt.xlabel("Timestamp", color="white")
#On check la stationarité de la série temporelle avec le test ADF

adfuller(df_train["sensor_00"],maxlag=50)
df_train["sensor_00"].shift(1).head()
sensor00_acf_plot = plot_acf((df_train["sensor_00"].shift(1)-df_train["sensor_00"]).dropna(), lags=50, title="ACF Sensor 00")
sensor00_pacf_plot = plot_pacf(df_train["sensor_00"], lags=50, title="PACF Sensor 00")
type(df_train.index)
rcParams['figure.figsize'] = 11, 9

decomposed_sensor00 = sm.tsa.seasonal_decompose(df_train["sensor_00"], freq=360)

figure = decomposed_sensor00.plot()
#resDiff = sm.tsa.arma_order_select_ic(df_train["sensor_00"], max_ar=10, max_ma=10, ic='aic', trend='c')

#print('ARMA(p,q) = ',resDiff["aic_min_order"],' is the best!')
model = SARIMAX(df_train["sensor_00"], order=(9,1,9))

results = model.fit()

results.summary()
results_plot = results.plot_diagnostics(figsize=(15,12))
df_train.index[0], df_train.index[-1]
tr_start, tr_end = '2018-04-01 00:00:00','2018-05-31 23:59:00'

tr_pred = '2018-06-10 00:00:00'

steps_to_predict = 5
forecast = results.forecast(steps_to_predict)
df_train["prediction"] = results.predict(start=70640,end=87840, dynamic=True)

df_train[["sensor_00","prediction"]].plot(figsize=(12,8))
df_train["prediction"] = results.predict(start=73640,end=87840, dynamic=True)

df_train[["sensor_00","prediction"]].plot(figsize=(12,8))