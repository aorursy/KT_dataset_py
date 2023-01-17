# pip install fbprophet
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import warnings

import pandas as pd

import matplotlib.pyplot as plt

# prophet model 

from fbprophet import Prophet

# prophet preformance

from fbprophet.diagnostics import cross_validation

from fbprophet.diagnostics import performance_metrics

from fbprophet.plot import plot_cross_validation_metric



# don't do this 

warnings.filterwarnings('ignore')

# "high resolution"

%config InlineBackend.figure_format = 'retina'



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
wmt = pd.read_csv('/kaggle/input/pwa_production consults_dt 2020-08-03T1509.csv')
wmt.info()
wmt.head(5)
wmt.columns = ['ds','y']
wmt.tail(5)
wmt.ds = pd.to_datetime(wmt.ds)
wmt.tail(5)
# frame up w/ grid

plt.figure(figsize=(16,4))

plt.grid(linestyle='-.')



# sketch in data

plt.plot(wmt.ds, wmt.y, 'b')



# set title & labels

plt.title('Consults', fontsize=18)

plt.ylabel('Visits ()', fontsize=13)

plt.xlabel('Time (month)', fontsize=13)



# display graph

plt.show()
# set prophet model 

prophet = Prophet(changepoint_prior_scale=0.15, daily_seasonality=True)
prophet.fit(wmt)
# build future dataframe for 1 year

build_forecast = prophet.make_future_dataframe(periods=365*1, freq='D')
# forecast future df w/ model

forecast = prophet.predict(build_forecast)
# plot forecasts

prophet.plot(forecast, xlabel='Date', ylabel='Visits')

plt.title('Consults')

# display graph

plt.show()
# tell us more about the forecast

prophet.plot_components(forecast)



# export forcast to csv

forecast.to_csv('consult_forecast.csv')
future_preds = forecast.loc[forecast.ds > '2020-01-31']

future_preds = future_preds[['ds','yhat','yhat_lower','yhat_upper']]

future_preds.sample(5)
# cross validate

wmt_cv = cross_validation(prophet, initial='30 days', period='60 days', horizon='90 days')
wmt_pm = performance_metrics(wmt_cv)
wmt_pm.tail(3)