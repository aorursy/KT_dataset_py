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
df_train = pd.read_csv('../input/train.csv', parse_dates=['date'], index_col=['date'])

df_train.info()
# choose one stroe to test with

sales2 = df_train[df_train.store == 2]['sales'].sort_index(ascending = True)



# the target var must be float type

df_train['sales'] = df_train['sales'].astype('float32')



# plot the total amount of sales, per day

sales2 = sales2.resample('D').sum()

sales2.plot(kind='area', figsize=(20,5), legend=True, alpha=.5)
# check for seasonality

import statsmodels.api as sm



# freq=the number of batches per period (year), if we use a weekly granularity, then freq=52

dec_a = sm.tsa.seasonal_decompose(sales2.resample('W').sum(), model = 'multiplicative', freq = 52)

dec_a.plot().show()
# use a rolling window and compute mean on the last "window" values (varied)

sales2_est30d = sales2.rolling(window=30).mean()

sales2_est60d = sales2.rolling(window=60).mean()

sales2_est90d = sales2.rolling(window=90).mean()
import matplotlib.pyplot as plt



# plot the week by week values: observed vs estimated for different values of window

plt.figure(figsize=(20, 5))

plt.plot(sales2.resample('W').sum(), color='Black', label="Observed", alpha=1)

plt.plot(sales2_est30d.resample('W').sum(), color='Red', label="Rolling mean trend, 30d", alpha=.5)

plt.plot(sales2_est60d.resample('W').sum(), color='Blue', label="Rolling mean trend, 60d", alpha=.5)

plt.plot(sales2_est90d.resample('W').sum(), color='Green', label="Rolling mean trend, 90d", alpha=.5)

plt.legend(loc="upper left")

plt.grid(True)

plt.show()
# import Prophet 

from fbprophet import Prophet



# reformat data

s = sales2.reset_index()

s.columns = ['ds', 'y']

s.head()
# fit a Prohet model. We assume there is no daily seasonality (or not interested in)

proph = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)

proph.fit(s[['ds','y']])
# make a void dataframe with future timestamps then compute predictions for it

fut = proph.make_future_dataframe(include_history=False, periods=12, freq = 'm') # 12 months

forecast = proph.predict(fut)
# plot all resulting components

proph.plot_components(forecast).show()
# another way to plot results

proph.plot(forecast).show()
sales2.head()
from pandas.plotting import autocorrelation_plot

autocorrelation_plot(sales2)
from statsmodels.tsa.arima_model import ARIMA

from matplotlib import pyplot



arim = ARIMA(sales2, order=(10,1,0)) # order=(p,d,q); p:AR param, d:differencing degree, q:windw size of MA

model_fit = arim.fit(disp=0)
#print(model_fit.summary())
# plot residual errors

residuals = pd.DataFrame(model_fit.resid)

residuals.plot()

# resid error density

residuals.plot.kde()
# generate a dateframe to forecast for

fut
# forecast

n_to_predict = 365

pred, err, bounds = model_fit.forecast(steps=n_to_predict) # predict for all dataset
from matplotlib.pyplot import plot

df = pd.DataFrame({'pred':pred})
full_df = pd.Series(pred.tolist(), index=pd.date_range('2018-01-01', freq='d', periods=n_to_predict).tolist())

full_df = sales2.append(full_df)
n = full_df.shape[0] - n_to_predict

fig, ax = plt.subplots(figsize=(20,5))

ax.plot(full_df[0:n], color='Black')

ax.plot(full_df[n-1:full_df.shape[0]], color='Red')

plt.show()