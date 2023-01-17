import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from datetime import datetime

from pytz import timezone

import pytz



%matplotlib inline



plt.style.use('fivethirtyeight')

train = pd.read_csv('../input/SolarPrediction.csv')
train.head()
train.shape
train.columns
train.dtypes
#hawaii = timezone('Pacific/Honolulu')



# Creamos una copia del original

train_origial = train.copy()

df = train.copy()



train_origial.index = pd.to_datetime(df['UNIXTime'], unit='s')

#df.index= df.index.tz_localize(pytz.utc).tz_convert(hawaii)



train['DateTime'] = train_origial.index

train_origial['DateTime'] = train_origial.index 

train.head()
train_radiation = train.drop(['UNIXTime', 'Data', 'Time', 'Temperature','TimeSunRise', 'TimeSunSet',

                         'Pressure', 'Humidity', 'WindDirection(Degrees)', 'Speed' ], axis=1)
for i in (train_radiation, train_origial):

    i['year'] = i.DateTime.dt.year

    i['month'] = i.DateTime.dt.month

    i['day'] = i.DateTime.dt.day

    i['Hour'] = i.DateTime.dt.hour
train_radiation['Day of week'] = train_radiation['DateTime'].dt.dayofweek

temp_rad = train_radiation['DateTime']
# Funcion para saber si es fin de semana o no, poco relevante ...

def applyer(row):

    if row.dayofweek == 5 or row.dayofweek == 6:

        return 1

    else:

        return 0

temp2 = train_radiation['DateTime'].apply(applyer)

train_radiation['weekend'] = temp2
train_radiation.index = train_radiation['DateTime']
train_radiation.head()
df_rad = train_radiation.drop('DateTime', 1)

ts = df_rad['Radiation']

plt.figure(figsize= (20,5))

plt.title('Radiation vs Time')

plt.xlabel('Time (Year-Month-Day))')

plt.ylabel('Radiation level')

plt.plot(ts)
train_radiation['Date']=pd.to_datetime(train_radiation.DateTime).dt.strftime('%Y-%m-%d')

train_radiation.index = train_radiation.Date

train_radiation.head()
train_radiation.groupby('month')['Radiation'].mean().plot.bar(figsize = (20,5),

                                                              title = 'Monthly Average Radiation',

                                                              fontsize = 14)
temp = train_radiation.groupby(['day'])['Radiation'].mean()

temp.plot(figsize = (20,5), title = "Average per day radiation Month", fontsize = 14)
temp = train_radiation.groupby(['day', 'Hour'])['Radiation'].mean()

temp.plot(figsize = (20,5), title = "Average Radiation per Daily, Hour", fontsize = 14)
temp = train_radiation.groupby(['Hour'])['Radiation'].mean()

temp.plot(figsize = (20,5), title = "Average Radiation per Hour", fontsize = 14)
train_radiation.groupby('Day of week')['Radiation'].mean().plot.bar(figsize = (20,6),

                                                                   title = 'Average radiation per day per week')
train_radiation['Timestamp'] = pd.to_datetime(train_radiation.DateTime, format = '%d-%m-%y %H:%M')

train_radiation.index = train_radiation.Timestamp



#Hourly

hourly = train_radiation.resample('H').mean()



#Daily

daily = train_radiation.resample('D').mean()



#Weekly

weekly = train_radiation.resample('W').mean()

    

#Monthly

monthly = train_radiation.resample('M').mean()
ig,axs = plt.subplots(4,1)



hourly.Radiation.plot(figsize = (15,8), title = "Hourly", fontsize = 14, ax = axs[0])

daily.Radiation.plot(figsize = (15,8), title = "Daily", fontsize = 14, ax = axs[1])

weekly.Radiation.plot(figsize = (15,8), title = "Weekly", fontsize = 14, ax = axs[2])

monthly.Radiation.plot(figsize = (15,8), title = "Monthly", fontsize = 14, ax = axs[3])

plt.tight_layout()
From = '2016-10-01'

To   = '2016-12-01'



hourly = hourly.loc[From:To,:]

daily = daily.loc[From:To,:]

weekly = weekly.loc[From:To,:] 

monthly = monthly.loc[From:To,:] 



ig,axs = plt.subplots(4,1)

hourly.Radiation.plot(figsize = (15,8), title = "Hourly", fontsize = 14, ax = axs[0])

daily.Radiation.plot(figsize = (15,8), title = "Daily", fontsize = 14, ax = axs[1])

weekly.Radiation.plot(figsize = (15,8), title = "Weekly", fontsize = 14, ax = axs[2])

monthly.Radiation.plot(figsize = (15,8), title = "Monthly", fontsize = 14, ax = axs[3])

plt.tight_layout()
from statsmodels.tsa.stattools import adfuller



def test_stationarity(df, ts):

    # Determining rolling statics

    rolmean = df[ts].rolling(window = 12, center = False).mean()

    rolstd = df[ts].rolling(window = 12, center = False).std()

    

    # Plot rolling statistics

    orig = plt.plot(df[ts], color = 'blue', label = 'Original')

    mean = plt.plot(rolmean, color = 'red' , label = 'Promedio')

    std = plt.plot(rolstd, color = 'black', label = 'Desviacion Estandar')

    

    plt.legend(loc = 'best')

    plt.title('Promedio y Desviacion Estandar para %s' %(ts))

    plt.xticks(rotation = 45)

    plt.show(block = False)

    plt.close()

    

    # Perform Dickey-Fuller test:

    # Null Hypothesis (H_0): time series is not stationary

    # Alternate Hypothesis (H_1): time series is stationary

    

    print('Results of Dickey-Fuller Test:')

    dftest = adfuller(df[ts], autolag='AIC')

    dfoutput = pd.Series(dftest[0:4],

                         index = ['Test Statistic',

                                  'p-value',

                                  '# Lags Used',

                                  'Number of Observations Used'])

    for key, value in dftest[4].items():

        dfoutput['Critical Value (%s)' %key] = value

    print(dfoutput)
test_stationarity(df = train_radiation, ts = 'Radiation')
from sklearn.metrics import mean_squared_error

from math import sqrt

import statsmodels.api as sm

from statsmodels.tsa.api import Holt, ExponentialSmoothing, SimpleExpSmoothing

from statsmodels.tsa.stattools import acf, pacf

from statsmodels.tsa.arima_model import ARIMA
_train = hourly.loc['2016-10-02':'2016-11-13',:]

valid = hourly.loc['2016-11-14': '2016-11-28',:]
_train.head()
valid.head()
_train.Radiation.plot(figsize=(25,5), title = 'Radiacion Diaria', fontsize=14, label='Train')

valid.Radiation.plot(figsize=(25,5), title = 'Radiacion Diaria', fontsize=14, label='Valid')

plt.xlabel('DateTime')

plt.ylabel('Radiation')

plt.legend(loc = 'best')
plt.style.use('default')

plt.figure(figsize = (16,8))

sm.tsa.seasonal_decompose(_train.Radiation).plot()

result = sm.tsa.stattools.adfuller(_train.Radiation)

plt.show()
dd = np.asarray(_train.Radiation)

y_hat =valid.copy()

y_hat['naive'] = dd[len(dd)- 1]

plt.figure(figsize = (25,5))

plt.plot(_train.index, _train['Radiation'],label = 'Train')

plt.plot(valid.index, valid['Radiation'], label = 'Validation')

plt.plot(y_hat.index, y_hat['naive'],  label = 'Naive')

plt.legend(loc = 'best')

plt.tick_params(axis = 'x', rotation = 45)
rmse = sqrt(mean_squared_error(valid['Radiation'], y_hat['naive']))

rmse
y_hat_holt = valid.copy()

fit1 = Holt(np.asarray(_train['Radiation'])).fit(smoothing_level = 0.01, smoothing_slope = 0.1)

y_hat_holt['Holt_linear'] = fit1.forecast(len(valid))

plt.style.use('fivethirtyeight')

plt.figure(figsize=(25,5))

plt.plot(_train.index, _train['Radiation'],label = 'Train')

plt.plot(valid.index, valid['Radiation'], label = 'Validation')

plt.plot(y_hat.index, y_hat_holt['Holt_linear'], label = 'Holt Linear')

plt.legend(loc='best')
rmse = sqrt(mean_squared_error(valid['Radiation'],  y_hat_holt.Holt_linear))

rmse
y_hat_avg2 = valid.copy()

fit2 = SimpleExpSmoothing(np.asarray(_train['Radiation'])).fit(smoothing_level=0.02,optimized=False)

y_hat_avg2['SES'] = fit2.forecast(len(valid))

plt.figure(figsize=(25,5))

plt.plot(_train['Radiation'], label='Train')

plt.plot(valid['Radiation'], label='Test')

plt.plot(y_hat_avg2['SES'], label='SES')

plt.legend(loc='best')

plt.show()
rms = sqrt(mean_squared_error(valid.Radiation, y_hat_avg2.SES))

print("Error: ", rms)
y_hat_avg = valid.copy()

fit1 = ExponentialSmoothing(np.asarray(_train['Radiation']), seasonal_periods=4, trend = 'add', seasonal= 'add').fit()

y_hat_avg['Holt_Winter'] = fit1.forecast(len(valid))

plt.figure(figsize = (25,5))

plt.plot(_train.index, _train['Radiation'],label = 'Train')

plt.plot(valid.index, valid['Radiation'], label = 'Validation')

plt.plot(y_hat_avg.index, y_hat_avg['Holt_Winter'], label = 'Holt_Winter')

plt.legend(loc = 'best')
rms = sqrt(mean_squared_error(valid.Radiation, y_hat_avg.Holt_Winter))

print("error: ", rms)
def plot_acf_pacf(df, ts):

  """

  Plot auto-correlation function (ACF) and partial auto-correlation (PACF) plots

  """

  f, (ax1, ax2) = plt.subplots(2,1, figsize = (10, 5)) 



  #Plot ACF: 



  ax1.plot(lag_acf)

  ax1.axhline(y=0,linestyle='--',color='gray')

  ax1.axhline(y=-1.96/np.sqrt(len(df[ts])),linestyle='--',color='gray')

  ax1.axhline(y=1.96/np.sqrt(len(df[ts])),linestyle='--',color='gray')

  ax1.set_title('Autocorrelation Function for %s' %(ts))



  #Plot PACF:

  ax2.plot(lag_pacf)

  ax2.axhline(y=0,linestyle='--',color='gray')

  ax2.axhline(y=-1.96/np.sqrt(len(df[ts])),linestyle='--',color='gray')

  ax2.axhline(y=1.96/np.sqrt(len(df[ts])),linestyle='--',color='gray')

  ax2.set_title('Partial Autocorrelation Function for %s' %(ts))

  

  plt.tight_layout()

  plt.show()

  plt.close()

  

  return
lag_acf = acf(np.array(_train['Radiation']), nlags = 20)

lag_pacf = pacf(np.array(_train['Radiation']), nlags = 20, method='ols')



plot_acf_pacf(df = _train, ts = 'Radiation')
fit2 = sm.tsa.statespace.SARIMAX(_train.Radiation, order=(1,0,1),seasonal_order=(1,1,0,12), trend='ct')

res = fit2.fit()

y_hat_avg['SARIMA'] = res.predict(start="2016-11-14", end="2016-11-29", dynamic=True)

plt.figure(figsize=(20,5))

plt.plot( _train['Radiation'], label='Train')

plt.plot(valid['Radiation'], label='Test')

plt.plot(y_hat_avg['SARIMA'], label='SARIMA')

plt.legend(loc='best')
res.summary()
rms = sqrt(mean_squared_error(valid.Radiation, y_hat_avg['SARIMA']))

print('Error:', rms)
model = ARIMA(_train.Radiation, order=(1, 0, 1))  

results_MA = model.fit()  

plt.plot(_train.Radiation)

plt.plot(results_MA.fittedvalues, color='red')
results_MA.summary()
def do_lstm_model(df, 

                  ts, 

                  look_back, 

                  epochs, 

                  type_ = None, 

                  train_fraction = 0.67):

  """

   Create LSTM model

   Source: https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

  """

  # Import packages

  import numpy

  import matplotlib.pyplot as plt

  from pandas import read_csv

  import math

  from keras.models import Sequential

  from keras.layers import Dense

  from keras.layers import LSTM

  from sklearn.preprocessing import MinMaxScaler

  from sklearn.metrics import mean_squared_error



  # Convert an array of values into a dataset matrix

  def create_dataset(dataset, look_back=1):

    """

    Create the dataset

    """

    dataX, dataY = [], []

    for i in range(len(dataset)-look_back-1):

      a = dataset[i:(i+look_back), 0]

      dataX.append(a)

      dataY.append(dataset[i + look_back, 0])

    return numpy.array(dataX), numpy.array(dataY)



  # Fix random seed for reproducibility

  numpy.random.seed(7)



  # Get dataset

  dataset = df[ts].values

  dataset = dataset.astype('float32')



  # Normalize the dataset

  scaler = MinMaxScaler(feature_range=(0, 1))

  dataset = scaler.fit_transform(dataset.reshape(-1, 1))

  

  # Split into train and test sets

  train_size = int(len(dataset) * train_fraction)

  test_size = len(dataset) - train_size

  train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

  

  # Reshape into X=t and Y=t+1

  look_back = look_back

  trainX, trainY = create_dataset(train, look_back)

  testX, testY = create_dataset(test, look_back)

  

  # Reshape input to be [samples, time steps, features]

  if type_ == 'regression with time steps':

    trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

    testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))

  elif type_ == 'stacked with memory between batches':

    trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

    testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))

  else:

    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

  

  # Create and fit the LSTM network

  batch_size = 1

  model = Sequential()

  

  if type_ == 'regression with time steps':

    model.add(LSTM(4, input_shape=(look_back, 1)))

  elif type_ == 'memory between batches':

    model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))

  elif type_ == 'stacked with memory between batches':

    model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))

    model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))

  else:

    model.add(LSTM(4, input_shape=(1, look_back)))

  

  model.add(Dense(1))

  model.compile(loss='mean_squared_error', optimizer='adam')



  if type_ == 'memory between batches' or type_ == 'stacked with memory between batches':

    for i in range(100):

      model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)

      model.reset_states()

  else:

    model.fit(trainX, 

              trainY, 

              epochs = epochs, 

              batch_size = 1, 

              verbose = 2)

  

  # Make predictions

  if type_ == 'memory between batches' or type_ == 'stacked with memory between batches':

    trainPredict = model.predict(trainX, batch_size=batch_size)

    testPredict = model.predict(testX, batch_size=batch_size)

  else:

    trainPredict = model.predict(trainX)

    testPredict = model.predict(testX)

  

  # Invert predictions

  trainPredict = scaler.inverse_transform(trainPredict)

  trainY = scaler.inverse_transform([trainY])

  testPredict = scaler.inverse_transform(testPredict)

  testY = scaler.inverse_transform([testY])

  

  # Calculate root mean squared error

  trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))

  print('Train Score: %.2f RMSE' % (trainScore))

  testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))

  print('Test Score: %.2f RMSE' % (testScore))

  

  # Shift train predictions for plotting

  trainPredictPlot = numpy.empty_like(dataset)

  trainPredictPlot[:, :] = numpy.nan

  trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

  

  # Shift test predictions for plotting

  testPredictPlot = numpy.empty_like(dataset)

  testPredictPlot[:, :] = numpy.nan

  testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

  

  # Plot baseline and predictions

  plt.plot(scaler.inverse_transform(dataset))

  plt.plot(trainPredictPlot)

  plt.plot(testPredictPlot)

  plt.show()

  plt.close()

  

  return
# LSTM Network for Regression

do_lstm_model(df = train_radiation, 

              ts = 'Radiation', 

              look_back = 1, 

              epochs = 5)