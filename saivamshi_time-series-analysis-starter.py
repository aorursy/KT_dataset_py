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
#This data is a time series of advance retail sales: Clothing and clothing accessory stores

cloth = pd.read_csv("/kaggle/input/advance-retail-sales-time-series-collection/RSCCASN.csv",parse_dates=["date"],index_col="date")

cloth = cloth.drop(["realtime_start","realtime_end"],axis=1)

cloth.head()
cloth.tail()
import matplotlib.pyplot as plt

cloth.plot()

plt.show()
#Checking the stationarity - Augmented Dickey-Fuller Test(ADF Test)



from statsmodels.tsa.stattools import adfuller



adf_stat = adfuller(cloth['value'])

print(adf_stat)
cloth["2013":"2014"].plot()

plt.show()
#Removing stationarity

cloth_yoy = cloth.pct_change(12).dropna()

cloth_yoy.head(5)
cloth_yoy.plot()
print(adfuller(cloth_yoy['value']))
cloth_yoy2 = cloth_yoy.diff().dropna()

cloth_yoy2.plot()
print(adfuller(cloth_yoy2['value']))
#Set the frequency to Days

cloth_yoy2 = cloth_yoy2.asfreq("MS")
#Splitting into train and test

train = cloth_yoy2.loc[:"2016"]

test = cloth_yoy2.loc["2017":]

test.head()
#Fitting a simple AR model

from statsmodels.tsa.statespace.sarimax import SARIMAX

model10 = SARIMAX(train,order=(1,0,1),trend="c")

results = model10.fit()

print(results.summary())
#One step ahead Forecasting

forecast = results.get_prediction(start=-24)

mean_forecast = forecast.predicted_mean

conf_int = forecast.conf_int()

fig,ax = plt.subplots(figsize=(18,6))

train.plot(ax=ax,label="observed")

mean_forecast.plot(ax=ax,color="r",label="predicted")

ax.fill_between(conf_int.index,conf_int["lower value"],conf_int["upper value"],color="pink")

ax.legend()

fig.show()
#Dynamic Forecasting in the past

forecast = results.get_prediction(start=-24,dynamic=True)

mean_forecast = forecast.predicted_mean

conf_int = forecast.conf_int()

fig,ax = plt.subplots(figsize=(18,6))

train.plot(ax=ax,label="observed")

mean_forecast.plot(ax=ax,color="r",label="predicted")

ax.fill_between(conf_int.index,conf_int["lower value"],conf_int["upper value"],color="pink")

ax.legend()

fig.show()
#Dynamic Forecasting into the future

forecast = results.get_forecast(steps=36)

mean_forecast = forecast.predicted_mean

conf_int = forecast.conf_int()

fig,ax = plt.subplots(figsize=(18,6))

train.plot(ax=ax,label="observed")

mean_forecast.plot(ax=ax,color="r",label="predicted")

ax.fill_between(conf_int.index,conf_int["lower value"],conf_int["upper value"],color="pink")

ax.legend()

fig.show()
#ARMAX model

trainx = train.copy()

trainx["month"] = trainx.index.month

trainx.head()
modelx = SARIMAX(trainx["value"],order=(1,0,1),exog=trainx["month"])

resultsx = modelx.fit()

print(resultsx.summary())
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

fig, ax = plt.subplots(2,1,figsize=(18,10))

plot_acf(cloth_yoy2,lags=10,zero=False,ax=ax[0])

plot_pacf(cloth_yoy2,lags=10,zero=False,ax=ax[1])
aic_dict = {}

for p in range(0,4):

    for q in range(0,4):

        model = SARIMAX(train,order=(p,0,q),trend="c")

        res = model.fit(maxiter=500)

        print(p,q,res.aic,res.bic)
#ARMA(2,3) process



mod23 = SARIMAX(train,order=(2,0,3),trend="c")

res23 = mod23.fit(maxiter=500)



forecast = res23.get_forecast(steps=36)

mean_forecast = forecast.predicted_mean

conf_int = forecast.conf_int()

fig,ax = plt.subplots(figsize=(18,6))

train.plot(ax=ax,label="observed",color="blue")

test.plot(ax=ax,color="blue")

mean_forecast.plot(ax=ax,color="r",label="predicted")

ax.fill_between(conf_int.index,conf_int["lower value"],conf_int["upper value"],color="pink")

ax.legend()

fig.show()
print("Mean absolute error: {}".format(np.mean(np.abs(res23.resid))))
res.plot_diagnostics(figsize=(15,8))

plt.show()
print(res.summary())
#Seasonal decomposition

from statsmodels.tsa.seasonal import seasonal_decompose

decomp = seasonal_decompose(cloth["value"],period=12)

decomp.plot()

plt.show()
#Seasonal differencing and normal differencing

cloth_diff = cloth.diff(1).diff(12).dropna()

cloth_diff.plot()

plt.show()
#Seasonal ACF and PACF

lags = [12, 24, 36, 48, 60]

fig, (ax1, ax2) = plt.subplots(2,1,figsize=(18,8))

plot_acf(cloth_diff,lags=lags,zero=False,ax=ax1)

plot_pacf(cloth_diff,lags=lags,zero=False,ax=ax2)
#SARIMAX

train_full = cloth.loc[:"2016"]

test_full = cloth.loc["2017":]



sarima_mod = SARIMAX(train_full,order=(2,1,3),seasonal_order=(0,1,0,12))

sarima_res = sarima_mod.fit(maxiter=500)

print(sarima_res.summary())
import pmdarima as pm

results = pm.auto_arima(cloth,maxiter=500,seasonal=True,m=12,information_criterion="aic",trace=True,error_action="ignore")

print(results.summary())
results.plot_diagnostics()
sarima_mod = SARIMAX(train_full,order=(1,0,2),seasonal_order=(2,1,2,12))

sarima_res = sarima_mod.fit(maxiter=500)

print(sarima_res.summary())
forecast = sarima_res.get_forecast(steps=60)

mean_forecast = forecast.predicted_mean

conf_int = forecast.conf_int()

fig,ax = plt.subplots(figsize=(18,6))

train_full.plot(ax=ax,label="observed",color="blue")

test_full.plot(ax=ax,color="blue")

mean_forecast.plot(ax=ax,color="r",label="predicted")

ax.fill_between(conf_int.index,conf_int["lower value"],conf_int["upper value"],color="pink")

ax.legend()

fig.show()
print(mean_forecast.iloc[-1])

print(conf_int.iloc[-1])