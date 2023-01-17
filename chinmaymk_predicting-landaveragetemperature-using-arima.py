# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
def parser(x):

	return pd.datetime.strptime(x, '%Y-%m-%d')

 

temperatures = pd.read_csv('../input/GlobalTemperatures.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
temperatures.LandAverageTemperature.plot()
resample = temperatures.resample('AS')

yearly = resample.mean()
yearly[['LandAverageTemperature']].plot(kind='line')
from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(yearly.LandAverageTemperature, order=(2,1,0))

arima_fit = model.fit()

plt.plot(arima_fit.fittedvalues)

plt.plot(yearly.LandAverageTemperature)
from fbprophet import Prophet
pmodel = Prophet()
subset = yearly.reset_index()[['dt', 'LandAverageTemperature']]

subset.rename(columns={"dt": "ds", "LandAverageTemperature": "y"}, inplace=True)

subset.head()
pmodel.fit(subset)

future = pmodel.make_future_dataframe(periods=30, freq='12M')

forecast = pmodel.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
pmodel.plot(forecast)
# variant 1

m2 = Prophet(growth='linear', 

             seasonality_prior_scale=1, 

             yearly_seasonality=False, 

             weekly_seasonality=False)

m2.fit(subset)

forecast2 = m2.predict(future)

m2.plot(forecast2)
# variant 2

m2 = Prophet(growth='linear', 

             seasonality_prior_scale=15, 

             yearly_seasonality=False, 

             weekly_seasonality=False)

m2.fit(subset)

forecast2 = m2.predict(future)

m2.plot(forecast2)
subset = yearly.reset_index()[['dt', 'LandAverageTemperature']]

subset.rename(columns={'dt': 'ds', 'LandAverageTemperature': 'y'}, inplace=True)

subset.head()
prophet = Prophet(yearly_seasonality=False, weekly_seasonality=False)

prophet.fit(subset)

future = prophet.make_future_dataframe(periods=25, freq='12M')
forecast = prophet.predict(future)


prophet.plot(forecast)

plt.axvline(x='2015-01-31')
print(yearly.iloc[-1]['LandAverageTemperature'])

print(forecast[forecast.ds == '2040-01-31'].iloc[0]['yhat'])

print(forecast[forecast.ds == '2040-01-31'].iloc[0]['yhat_upper'])