import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# See https://facebook.github.io/prophet/docs/quick_start.html
from fbprophet import Prophet
# See https://facebook.github.io/prophet/docs/diagnostics.html
from fbprophet.diagnostics import cross_validation

# See http://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# read prices table into Pandas DataFrame
df = pd.read_csv('../input/prices-split-adjusted.csv')

# pick a random stock
ticker = np.random.choice(df.symbol.unique())
df = df.loc[df['symbol'] == ticker]

# split into train/test 90/10
df_size = len(df)
train_size = round(df_size*90/100)
test_size = round(df_size*10/100)

df_train = df[:train_size]
df_test = df[-test_size:]

print(ticker, df_size, train_size, test_size)
# init Prophet dataframe
train = pd.DataFrame()
# convert to Prophet required form
train[['ds', 'y']] = df_train[['date', 'close']]

# Log-transform the target variable, as per Prophet tutorial. Why do they do this?
# train['y'] = np.log(train['y'])
# Init & fit model
model = Prophet()
model.fit(train)
# Forecast the future up until the end of our test data
# TODO: normalize weekends where data is missing in test_data, or length will not match
forecast_frame = model.make_future_dataframe(periods=test_size)
# last 5 entries
forecast_frame.tail()
# Forecast
forecast = model.predict(forecast_frame)
# last 5 entries: `yhat` is the actual prediction
# see Prophet documentation for the rest
forecast.tail()
# Run prediction for next 30 days every 30 days, starting after training on the first 5 years of data
validation = cross_validation(model, horizon = '30 days', period = '30 days', initial = '1825 days')
# last 5 entries
validation.tail()
# Manually compute mean error percentage
mean_error = np.mean(np.abs((validation['y'] - validation['yhat']) / validation['y'])) * 100
mean_error
mean_absolute_error(validation['y'], validation['yhat'])
mean_squared_error(validation['y'], validation['yhat'])
model.plot(forecast)
