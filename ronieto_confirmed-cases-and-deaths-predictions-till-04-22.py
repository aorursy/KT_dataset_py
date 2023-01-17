# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from fbprophet import Prophet
df2 = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv', sep = ',')
## I will use train set only

df2.tail()
confirmed = df2.groupby('Date').sum()['ConfirmedCases'].reset_index()

death = df2.groupby('Date').sum()['Fatalities'].reset_index()
confirmed.columns = ['ds','y']

confirmed['ds'] = pd.to_datetime(confirmed['ds'])
m = Prophet(interval_width=0.97)

m.fit(confirmed)

future = m.make_future_dataframe(periods=29)

future_confirmed = future.copy() 

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
confirmed_cases = m.plot(forecast)
forecast_components = m.plot_components(forecast)
forecast1=pd.DataFrame(forecast)

## These are the predictions for Confirmed Covid-19 cases until 2020 April 22

forecastC= forecast1[['ds', 'yhat']]

forecastC.columns = [['ForecastId', 'ConfirmedCases']]

death.columns = ['ds','y']

death['ds'] = pd.to_datetime(death['ds'])
m = Prophet(interval_width=0.97)

m.fit(death)

future = m.make_future_dataframe(periods=29)

future_deaths = future.copy() 

future.tail()
forecastD = m.predict(future)

forecastD[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
death_predict = pd.DataFrame(forecastD)
## These are the predictions for Deaths by Covid-19 cases until 2020 April 24



forecastDeath= death_predict[['ds', 'yhat']]

forecastDeath.columns = [['ForecastId', 'Fatalities']]

death_forecast = m.plot(forecastD)
forecastD_components = m.plot_components(forecastD)

forecastD_components
submission = pd.merge(forecastC, forecastDeath, how='inner')
submission.to_csv('submission.csv', index=False)
