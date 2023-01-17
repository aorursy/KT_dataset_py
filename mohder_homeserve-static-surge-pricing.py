import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os



# registering pandas converers to allow plotting with pd conveniently

#pd.plotting.register_matplotlib_converters()
df_train = pd.read_csv('../input/homeserve-gas-rates/3ygas.csv', parse_dates=['Date'], index_col=['Date'], sep=';', decimal=",")

df_train.info()
# display a bunch of data

df_train.head()
# plot data

df_train.plot(figsize=(20,8))
# choose one variable to test and play with, let it be "base"

ss = df_train['base'].sort_index(ascending = True)



# the target var must be float type

df_train['base'] = df_train['base'].astype('float32')



# plot the total rates, per day

ss = ss.resample('D').sum()

ss.plot(kind='area', figsize=(20,5), legend=True, alpha=.5)
# check for seasonality

import statsmodels.api as sm

        

# freq=the number of batches per period (year), as we use a weekly granularity ('W'), then freq=52

dec_a = sm.tsa.seasonal_decompose(ss.resample('W').sum(), model = 'additive', freq = 52)

dec_a.plot().show()
# use a rolling window and compute mean on the last "window" values (varied)

ss_est30d = ss.rolling(window=30).mean()

ss_est60d = ss.rolling(window=60).mean()

ss_est90d = ss.rolling(window=90).mean()
import matplotlib.pyplot as plt



# plot the week by week values: observed vs estimated for different values of window

plt.figure(figsize=(20, 5))

plt.plot(ss.resample('W').sum(), color='Black', label="Observed", alpha=1)

plt.plot(ss_est30d.resample('W').sum(), color='Red', label="Prediction: Rolling mean trend, based on 30d", alpha=.5)

plt.plot(ss_est60d.resample('W').sum(), color='Blue', label="Prediction: Rolling mean trend, based on 60d", alpha=.5)

plt.plot(ss_est90d.resample('W').sum(), color='Green', label="Prediction: Rolling mean trend, based on 90d", alpha=.5)

plt.legend(loc="upper left")

plt.grid(True)

plt.show()
# import Prophet 

from fbprophet import Prophet



#pd.plotting.deregister_matplotlib_converters()



import imp  

imp.reload(pd)



# reformat data

s = ss.reset_index()

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
#!jupyter nbconvert --execute --to pdf __notebook_source__.ipynb

#!curl --upload-file __notebook_source__.pdf https://transfer.sh/notebook.pdf