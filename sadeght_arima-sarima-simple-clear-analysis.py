# Import libraries 

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



import os

print(os.listdir("../input"))
# Read our data

data = pd.read_csv('../input/IPG2211A2N.csv',index_col=0)

data.head()
# Change our data index from string to datetime

data.index = pd.to_datetime(data.index)

data.columns = ['Energy Production']

data.head()
# Import Plotly & Cufflinks libraries and run it in Offline mode

import plotly.offline as py

py.init_notebook_mode(connected=True)

py.enable_mpl_offline()



import cufflinks as cf

cf.go_offline()
# Now, plot our time serie

data.iplot(title="Energy Production Between Jan 1939 to May 2019")
# We'll use statsmodels to perform a decomposition of this time series

from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(data, model='multiplicative')



fig = result.plot()
py.iplot_mpl(fig)

# Try "py.plot_mpl(fig)" on your local Anaconda, it'll show greater plot than this one
!pip install pmdarima
# Kaggle doesn't have the up-to-date version of Scipy, so we need to upgrade it in order to use pmdarima library

!pip install --upgrade scipy
# The Pmdarima library for Python allows us to quickly perform this grid search 

from pmdarima import auto_arima
stepwise_model = auto_arima(data, start_p=1, start_q=1,

                           max_p=3, max_q=3, m=12,

                           start_P=0, seasonal=True,

                           d=1, D=1, trace=True,

                           error_action='ignore',  

                           suppress_warnings=True, 

                           stepwise=True)
print(stepwise_model.aic())
# For the Test: we'll need to chop off a portion of our latest data, say from 2016, Jan.

test = data.loc['2016-01-01':]



# Fore the Train: we'll train on the rest of the data after split the test portion

train = data.loc['1939-01-01':'2015-12-01']
stepwise_model.fit(train)
future_forecast = stepwise_model.predict(n_periods=41)

print(future_forecast)
future_forecast = pd.DataFrame(future_forecast, index=test.index, columns=['Prediction'])
pd.concat([test, future_forecast], axis=1).iplot()
pd.concat([data,future_forecast],axis=1).iplot()
stepwise_model.fit(data)
future_forecast_1year = stepwise_model.predict(n_periods=13)
# For a year forecasting, we need 13 rows from 2019-05-01 to 2020-05-01

next_year = [pd.to_datetime('2019-05-01'),

            pd.to_datetime('2019-06-01'),

            pd.to_datetime('2019-07-01'),

            pd.to_datetime('2019-08-01'),

            pd.to_datetime('2019-09-01'),

            pd.to_datetime('2019-10-01'),

            pd.to_datetime('2019-11-01'),

            pd.to_datetime('2019-12-01'),

            pd.to_datetime('2020-01-01'),

            pd.to_datetime('2020-02-01'),

            pd.to_datetime('2020-03-01'),

            pd.to_datetime('2020-04-01'),

            pd.to_datetime('2020-05-01')]
future_forecast_1year = pd.DataFrame(future_forecast_1year, index=next_year, columns=['Prediction'])
pd.concat([data,future_forecast_1year],axis=1).iplot()