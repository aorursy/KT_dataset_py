#import libraries



import warnings

import datetime

import itertools

import numpy as np

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

plt.style.use('fivethirtyeight')

import pandas as pd

import statsmodels.api as sm

import matplotlib

matplotlib.rcParams['axes.labelsize'] = 14

matplotlib.rcParams['xtick.labelsize'] = 12

matplotlib.rcParams['ytick.labelsize'] = 12

matplotlib.rcParams['text.color'] = 'k'
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data = pd.read_csv('/kaggle/input/sea-level-change/sea_levels_2015.csv')

data.shape
data.head()
#check for the null values

data.isnull().sum()
#formatting the date column correctly

data.Time=data.Time.apply(lambda x:datetime.datetime.strptime(x, '%Y-%m-%d'))

# check

print(data.info())
data.head()
ts=data.groupby(["GMSL"])["GMSL"].sum()

ts.astype('float')

plt.figure(figsize=(14,8))

plt.title('Global Average Absolute Sea Level Change')

plt.xlabel('Time')

plt.ylabel('Sea Level Change')

plt.plot(ts);
plt.figure(figsize=(14,6))

plt.plot(ts.rolling(window=12,center=False).mean(),label='Rolling Mean');

plt.plot(ts.rolling(window=12,center=False).std(),label='Rolling Deviation');

plt.legend();
# Additive model

res = sm.tsa.seasonal_decompose(ts.values,freq=12,model="additive")

#plt.figure(figsize=(16,12))

fig = res.plot()

#fig.show()
import statsmodels.api as sm

import warnings

warnings.filterwarnings("ignore")

mod = sm.tsa.statespace.SARIMAX(ts.values,

                                order = (2, 0, 4),

                                seasonal_order = (3, 1, 2, 12),

                                enforce_stationarity = False,

                                enforce_invertibility = False)

results = mod.fit()

results.plot_diagnostics(figsize=(14,12))

plt.show()
from fbprophet import Prophet



ts = data.rename(columns={'Time':'ds', 'GMSL':'y','GMSL uncertainty':'yhat'})



ts.columns=['ds','y','yhat']

model1 = Prophet( yearly_seasonality=True) 

model1.fit(ts)
# predict for 1 year in the furure and MS - month start is the frequency

future = model1.make_future_dataframe(periods = 12, freq = 'MS')  

# now lets make the forecasts

forecast = model1.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
model1.plot(forecast)