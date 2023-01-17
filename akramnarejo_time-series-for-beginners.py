import pandas as pd

import pandas_datareader as web # it lets us download the stock prices 

from fbprophet import Prophet # Prophet framework for time series

from datetime import datetime

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
today = datetime.now() #returns the today's date and time

today
start = today.year-5 # defining the numbers of years of we need the data
#lets create a dataframe

df = web.get_data_yahoo('AMZN', start, today)
df.reset_index(inplace=True)
df = df[['Date','Close']]
# Renaming the columns as per model requirements

df = df.rename(columns={'Date':'ds', 'Close':'y'})
df.head()
model = Prophet()
model.fit(df)
# creating the dataframe for length of time we have to forecast. we are creating for next 1 year

future_dates = model.make_future_dataframe(periods=12, freq='M') # we have to define period according to frequency ( number of months or weeks or days )
future_dates.tail()
# lets forecast

forecast = model.predict(future_dates)
# Ploting the forecasted stock prices and for the framework has the builin plot method

model.plot(forecast, uncertainty=True);
model.plot_components(forecast);