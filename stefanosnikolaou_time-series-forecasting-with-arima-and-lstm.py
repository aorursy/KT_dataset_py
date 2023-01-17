import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Load data
df = pd.read_csv ('../input/dblp-2020-05-01-pubs-per-year/pubs.csv')

# Draw plot
plt.style.use('dark_background')
plot=df.plot(kind = 'bar', x = 'Year', y = 'Pubs', color = 'orange', figsize = (16,9))

# Format plot
plt.title('Records in DBLP', fontweight ='bold', color = 'orange')
plt.xlabel('Year', labelpad = 20, fontweight ='bold', color = 'orange')
plt.ylabel('Publications', labelpad = 20, fontweight ='bold', color = 'orange',)

# Draw horizontal axis lines
axes = plt.gca()
axes.get_legend().remove()
axes.yaxis.grid(color = 'grey', linestyle = 'dashed')

# Format x axis ticks
l=np.array(df['Year'])
plt.xticks(range(0, len(l), 5), l[::5], rotation = 0, fontweight ='bold')

# Format y axis ticks
plot.yaxis.set_major_formatter(ticker.EngFormatter())
plt.yticks(fontweight ='bold')

plt.show()


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

# Load data
data = pd.read_csv('../input/dblp-2020-05-01-pubs-per-year/pubs_cut.csv',parse_dates=[0])
con=data['Year']
data['Year']=pd.to_datetime(data['Year'])
data.set_index('Year', inplace=True)

# cCheck datatype of index
data.index

# Convert to time series:
ts = data['Pubs']

# Log transform time series
ts_log=np.log(ts)

# Create plots
decomposition=seasonal_decompose(ts_log)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
plt.figure(figsize = (16,9))
plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Perform Dickey-Fuller test:
print(
    'Results of Dickey-Fuller Test:')
dftest = adfuller(ts_log, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)' % key] = value
print(dfoutput)
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

# Load data
df = pd.read_csv ('../input/dblp-2020-05-01-pubs-per-year/pubs_cut.csv') 

# Set Year as index
df = df.set_index("Year")

# Create acf and pacf plots
plt.style.use('fivethirtyeight')
plt.figure(figsize = (16,9))
ax1=plt.subplot(211)
plot_acf(df, ax=plt.gca(), lags=20)
plt.title('Autocorrelation', color='black')
ax1.tick_params(axis='y', colors='black')
ax1.tick_params(axis='x', colors='black')
ax2=plt.subplot(212)
plot_pacf(df, ax=plt.gca(), lags=20)
ax2.tick_params(axis='y', colors='black')
ax2.tick_params(axis='x', colors='black')
plt.title('Partial Autocorrelation', color='black')
plt.axis(color='black')
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import statistics
from statsmodels.tools.eval_measures import rmse
from statsmodels.tsa.arima_model import ARIMA

warnings.filterwarnings("ignore")

# Insert data without years 1918 & 2020
df = pd.read_csv ('../input/dblp-2020-05-01-pubs-per-year/pubs_cut.csv', parse_dates=[0])

# Format 'Year' column as year type
df.Year = pd.to_datetime(df.Year).dt.year

# Build train and test set
train_data = df[:len(df)-12]
train_data = train_data.astype('float64')
test_data = df[len(df)-12:]
test_data = test_data.astype('float64')

# Build ARIMA model (2,1,0)
model = ARIMA(train_data['Pubs'], order=(2,1,0))
model_fit = model.fit(disp=0)
arima_pred = model_fit.predict(start = len(train_data), end = len(df)-1, typ="levels").rename("ARIMA Forecast")
print(model_fit.summary())

# Build plot
plt.style.use('fivethirtyeight')
train_data['Pubs'].plot(figsize = (16,9), legend=True, marker='o')
test_data['Pubs'].plot(figsize = (16,9), legend=True, marker='o')
plt.title("Time Series Forecasting with ARIMA", fontweight ='bold', color='black')
plt.xlabel('Year', fontweight ='bold', color='black')
plt.ylabel('Publications', fontweight ='bold', color='black')

# Format x axis ticks
plt.xlabel('Year')
l=np.array(df['Year'])
plt.xticks(range(0, len(l), 5), l[::5], rotation = 0, color='black')
plt.yticks(color='black')

# Add markers to predictions line
arima_pred.plot(legend = True, marker='o');

# Rename legend labels
L=plt.legend()
L.get_texts()[0].set_text('Actual publications')
L.get_texts()[1].set_text('Train Data')
L.get_texts()[2].set_text('Predicted Data')
plt.setp(L.get_texts(), color = 'black')

# Calculate test_data mean
x= statistics.mean(test_data['Pubs'])
print("Mean is:", x)

# Calculate RMSE
arima_rmse_error = rmse(test_data['Pubs'], arima_pred)
print(f'RMSE = {arima_rmse_error}')

plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics
from statsmodels.tools.eval_measures import rmse
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings

warnings.filterwarnings("ignore")

# Insert data without years 1918 & 2020
df = pd.read_csv ('../input/dblp-2020-05-01-pubs-per-year/pubs_cut.csv', parse_dates=[0])

# Format 'Year' column as year type
df.Year = pd.to_datetime(df.Year).dt.year

# Build train and test set
train_data = df[:len(df)-12]
train_data = train_data.astype('float64')
test_data = df[len(df)-12:]
test_data = test_data.astype('float64')

def arima(p,d,q):
    try:
        try:
            model = ARIMA(train_data['Pubs'], order=(p,d,q))
            model_fit = model.fit(disp=0)
            arima_pred = model_fit.predict(start=len(train_data), end=len(df) - 1, typ="levels").rename("ARIMA Forecast")
            arima_rmse_error = sqrt(mean_squared_error(test_data['Pubs'], arima_pred))
        except np.linalg.LinAlgError as e:
            arima_rmse_error=100000
    except ValueError as e:
        arima_rmse_error=100000
    return(arima_rmse_error)

for p in range (0,4):
    for d in range (0,3):
        for q in range  (0,7):
            arima_rmse_error=arima(p,d,q)
            if q == 0 and d == 0 and p == 0:
                rmse = arima_rmse_error
            if rmse > arima_rmse_error:
                rmse = arima_rmse_error
                P,D,Q=p,d,q

model = ARIMA(train_data['Pubs'], order=(P, D, Q))
model_fit = model.fit(disp=0)
arima_pred = model_fit.predict(start=len(train_data), end=len(df) - 1, typ="levels").rename("ARIMA Forecast")
arima_rmse_error = sqrt(mean_squared_error(test_data['Pubs'], arima_pred))
print(model_fit.summary())

# Calculate test_data mean
x= statistics.mean(test_data['Pubs'])
print("Mean is :", x)
plt.show()

# Show RMSE
print(f'RMSE = {arima_rmse_error}')

test_data["ARIMA"] = arima_pred 
# Build plot
plt.style.use('fivethirtyeight')
train_data['Pubs'].plot(figsize = (16,9), legend=True, marker='o')
test_data['Pubs'].plot(figsize = (16,9), legend=True, marker='o')
plt.title("Time Series Forecasting with ARIMA", fontweight ='bold', color='black')
plt.xlabel('Year', fontweight ='bold', color='black')
plt.ylabel('Publications', fontweight ='bold', color='black')

# Format x axis ticks
plt.xlabel('Year')
l=np.array(df['Year'])
plt.xticks(range(0, len(l), 5), l[::5], rotation = 0, color='black')
plt.yticks(color='black')

# Add markers to predictions line
arima_pred.plot(legend = True, marker='o');

# Rename legend labels
L=plt.legend()
L.get_texts()[0].set_text('Actual publications')
L.get_texts()[1].set_text('Train Data')
L.get_texts()[2].set_text('Predicted Data')
plt.setp(L.get_texts(), color = 'black')

plt.show()
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy
import pandas as pd

# datetime parsing function for loading the dataset

def parser(x):
   return datetime.strptime(x, '%Y')

# frame a sequence as a supervised learning problem

def timeseries_to_supervised(data, lag=1):
   df = DataFrame(data)
   columns = [df.shift(i) for i in range(1, lag+1)]
   columns.append(df)
   df = concat(columns, axis=1)
   df.fillna(0, inplace=True)
   return df

# create a differenced series

def difference(dataset, interval=1):
   diff = list()
   for i in range(interval, len(dataset)):
      value = dataset[i] - dataset[i - interval]
      diff.append(value)
   return Series(diff)

# invert differenced value

def inverse_difference(history, yhat, interval=1):
   return yhat + history[-interval]

# scale train and test data to [-1, 1]

def scale(train, test):
   # fit scaler
   scaler = MinMaxScaler(feature_range=(-1, 1))
   scaler = scaler.fit(train)
   # transform train
   train = train.reshape(train.shape[0], train.shape[1])
   train_scaled = scaler.transform(train)
   # transform test
   test = test.reshape(test.shape[0], test.shape[1])
   test_scaled = scaler.transform(test)
   return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value

def invert_scale(scaler, X, value):
   new_row = [x for x in X] + [value]
   array = numpy.array(new_row)
   array = array.reshape(1, len(array))
   inverted = scaler.inverse_transform(array)
   return inverted[0, -1]

# fit an LSTM network to training data

def fit_lstm(train, batch_size, nb_epoch, neurons):
   X, y = train[:, 0:-1], train[:, -1]
   X = X.reshape(X.shape[0], 1, X.shape[1])
   model = Sequential()
   model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
   model.add(Dense(1))
   model.compile(loss='mean_squared_error', optimizer='adam')
   for i in range(nb_epoch):
      model.fit(X, y, epochs=10, batch_size=batch_size, verbose=1, shuffle=False)
      model.reset_states()
   return model

# make a one-step forecast

def forecast_lstm(model, batch_size, X):
   X = X.reshape(1, 1, len(X))
   yhat = model.predict(X, batch_size=batch_size)
   return yhat[0,0]

# load dataset

series = read_csv('../input/dblp-2020-05-01-pubs-per-year/pubs_cut.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

# transform data to be stationary

raw_values = series.values
diff_values = difference(raw_values, 1)

# transform data to be supervised learning

supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

# split data into train and test-sets

train, test = supervised_values[0:-12], supervised_values[-12:]

# transform the scale of the data

scaler, train_scaled, test_scaled = scale(train, test)

# fit the model

lstm_model = fit_lstm(train_scaled, 1, 3, 4)

# forecast the entire training dataset to build up state for forecasting

train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)

# walk-forward validation on the test data

predictions = list()
for i in range(len(test_scaled)):
   # make one-step forecast
   X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
   yhat = forecast_lstm(lstm_model, 1, X)
   # invert scaling
   yhat = invert_scale(scaler, X, yhat)
   # invert differencing
   yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
   # store forecast
   predictions.append(yhat)
   expected = raw_values[len(train) + i + 1]
   print('Year=%d, Predicted=%f, Expected=%f' % (i+2008, yhat, expected))

# report performance

rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
print('Test RMSE: %.3f' % rmse)

# line plot of observed vs predicted

b = numpy.arange(1936, 2020, 1)
df1 = pd.DataFrame(b)
df2 = pd.DataFrame(raw_values)
df3 = pd.DataFrame(predictions)
for i in range (0,72):
   df3.loc[len(df3)] = numpy.nan
   df3 = df3.shift()
   df3.loc[0] = numpy.nan
print(df3)
pdlist=[df1,df2,df3]
df1.reset_index(drop=True, inplace=True)
df2.reset_index(drop=True, inplace=True)
df3.reset_index(drop=True, inplace=True)
df = pd.concat(pdlist, axis=1)
df.columns=['Year','Actual Values','Predicted Values']
print(df)

test_data['LSTM'] = lstm_predictions
# Draw plot

pyplot.style.use('fivethirtyeight')
plt.figure(figsize = (16,9))
pyplot.plot(df['Actual Values'], marker='o', label="Actual")
pyplot.plot(df['Predicted Values'], marker='o', label="LSTM Forecat")
pyplot.title("Time Series Forecasting with LSTM", fontweight ='bold', color = 'black')
pyplot.xlabel('Year', fontweight ='bold', color = 'black')
pyplot.ylabel('Publications', fontweight ='bold', color = 'black')
a=numpy.array(df['Year'])
pyplot.xticks(range(0, len(a), 5), a[::5], rotation = 0, color = 'black')
pyplot.yticks(color = 'black')
L=plt.legend()
L.get_texts()[0]
L.get_texts()[1]
plt.setp(L.get_texts(), color = 'black')

pyplot.show()
# Draw comparison plot
                  
pyplot.style.use('fivethirtyeight')
plt.figure(figsize = (16,9))
plt.plot(test_data['Year'], test_data["Pubs"], label="Actual",)
plt.plot(test_data['Year'], test_data["ARIMA"], linestyle="--", label="ARIMA")
plt.plot(test_data['Year'], test_data["LSTM"], linestyle="--", label="LSTM", color="purple")
pyplot.title("ARIMA vs LSTM", fontweight ='bold', color = 'black')
pyplot.xlabel('Year', fontweight ='bold', color = 'black')
pyplot.ylabel('Publications', fontweight ='bold', color = 'black')
pyplot.xticks(color = 'black')
pyplot.yticks(color = 'black')
L=plt.legend()
L.get_texts()[0]
L.get_texts()[1]
L.get_texts()[2]
plt.setp(L.get_texts(), color = 'black')

plt.show()
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy
import pandas as pd

# date-time parsing function for loading the dataset
def parser(x):
	return datetime.strptime(x, '%Y')

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = numpy.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=10, batch_size=batch_size, verbose=1, shuffle=False)
		model.reset_states()
	return model

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

# load dataset
series = read_csv('../input/dblp-2020-05-01-pubs-per-year/pubs.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)[1:]

# transform data to be stationary
raw_values = series.values
diff_values = difference(raw_values, 1)

# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

# split data into train and test-sets
train, test = supervised_values[0:-13], supervised_values[-13:]

# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)

# fit the model
lstm_model = fit_lstm(train_scaled, 1, 3, 4)
# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)

# walk-forward validation on the test data
predictions = list()
for i in range(len(test_scaled)):
	# make one-step forecast
	X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
	yhat = forecast_lstm(lstm_model, 1, X)
	# invert scaling
	yhat = invert_scale(scaler, X, yhat)
	# invert differencing
	yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
	# store forecast
	predictions.append(yhat)
	expected = raw_values[len(train) + i + 1]
	print('Year=%d, Predicted=%f, Expected=%f' % (i+2008, yhat, expected))

# report performance
rmse = sqrt(mean_squared_error(raw_values[-13:-1], predictions[-13:-1]))
print('Test RMSE: %.3f' % rmse)
# line plot of observed vs predicted

b = numpy.arange(1936, 2021, 1)
df1 = pd.DataFrame(b)
df2 = pd.DataFrame(raw_values)
df3 = pd.DataFrame(predictions)
for i in range (0,72):
	df3.loc[len(df3)] = numpy.nan
	df3 = df3.shift()
	df3.loc[0] = numpy.nan
pdlist=[df1,df2,df3]
df1.reset_index(drop=True, inplace=True)
df2.reset_index(drop=True, inplace=True)
df3.reset_index(drop=True, inplace=True)
df = pd.concat(pdlist, axis=1)
df.columns=['Year','Actual Values','Predicted Values']
print(df)
# Draw plot

pyplot.style.use('fivethirtyeight')
ax=df.plot.bar(x='Year', y='Actual Values', legend=True, figsize = (16,9))
df[:-12].plot(y='Actual Values', linestyle='-',ax=ax,color='red',  legend=True)
df.plot(y='Predicted Values', linestyle='-',ax=ax,color='green',  legend=True)
plt.title('Prediction For 2020 Publications',fontweight ='bold', color = 'black')
plt.xlabel('Year', fontweight ='bold', color='black')
plt.ylabel('Publications', fontweight ='bold', color='black')
a = numpy.arange(1936, 2022, 1)
plt.xticks(range(0, len(a), 5), a[::5], rotation = 30, color='black')
plt.yticks(color='black')
L=pyplot.legend()
L.get_texts()[2].set_text('Actual publications')
L.get_texts()[0].set_text('Train Data')
L.get_texts()[1].set_text('Predicted Data')
plt.setp(L.get_texts(), color = 'black')

pyplot.show()