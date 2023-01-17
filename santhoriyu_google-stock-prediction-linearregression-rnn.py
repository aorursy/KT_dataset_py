import pandas as pd

import numpy as np

pd.options.display.max_columns=30

from IPython.display import display, HTML, display_html

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

#Disply all the cell output

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = 'all'

import xgboost as xgb

from xgboost import plot_importance, plot_tree

from sklearn.metrics import mean_squared_error, mean_absolute_error

plt.style.use('fivethirtyeight')
df = pd.read_csv('../input/GOOG.csv', index_col=0, parse_dates=True)

df.head()
df[['Open','Close']].plot(figsize=(15, 5),style=['-','.'])
df['date'] = df.index

df['hour'] = df['date'].dt.hour

df['dayofweek'] = df['date'].dt.dayofweek

df['quarter'] = df['date'].dt.quarter

df['month'] = df['date'].dt.month

df['year'] = df['date'].dt.year

df['dayofyear'] = df['date'].dt.dayofyear

df['dayofmonth'] = df['date'].dt.day

df['weekofyear'] = df['date'].dt.weekofyear

# df['Price_After_Month']=df['Adj. Close'].shift(-30)
cols = ['Open', 'High', 'Low', 'Adj Close', 'date']

df.drop(cols,1,inplace=True)
df.head()
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, OneHotEncoder



X = df.drop('Close',1)

y = df['Close']
#Feature Importance

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=1, max_depth=10)

model.fit(X,y)

features = X.columns

importances = model.feature_importances_

indices = np.argsort(importances)[-9:]  # top 10 features

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()
X.drop(['Volume', 'hour', 'dayofweek', 'quarter', 'month', 'dayofyear',

       'dayofmonth', 'weekofyear'], 1, inplace=True)
SC1 = StandardScaler()

X = SC1.fit_transform(X)

X
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
#Linear Regression

from sklearn.linear_model import LinearRegression

Reg1 = LinearRegression()

Reg1.fit(X_train,y_train)

Reg1.score(X_test, y_test),mean_squared_error(y_test, Reg1.predict(X_test)), Reg1.score(X_train,y_train)
y_test[:10].values, Reg1.predict(X_test)[:10]
df2 = df['Close']

df2 = pd.DataFrame(df2)

df2 = df2.rename(columns = {'Close': 'ts'})

df2.head()
from statsmodels.tsa.stattools import adfuller

def test_stationarity(df, ts):

    """

    Test stationarity using moving average statistics and Dickey-Fuller test

    Source: https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/

    """

    

    # Determing rolling statistics

    rolmean = df[ts].rolling(window = 12, center = False).mean()

    rolstd = df[ts].rolling(window = 12, center = False).std()

    

    # Plot rolling statistics:

    plt.figure(figsize=(15,10))

    orig = plt.plot(df[ts], 

                    color = 'blue', 

                    label = 'Original')

    mean = plt.plot(rolmean, 

                    color = 'red', 

                    label = 'Rolling Mean')

    std = plt.plot(rolstd, 

                   color = 'black', 

                   label = 'Rolling Std')

    plt.legend(loc = 'best')

    plt.title('Rolling Mean & Standard Deviation for %s' %(ts))

    plt.xticks(rotation = 45)

    plt.show(block = False)

    plt.close()

    

    # Perform Dickey-Fuller test:

    # Null Hypothesis (H_0): time series is not stationary

    # Alternate Hypothesis (H_1): time series is stationary

    print ('Results of Dickey-Fuller Test:')

    dftest = adfuller(df[ts], 

                      autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], 

                         index = ['Test Statistic',

                                  'p-value',

                                  '# Lags Used',

                                  'Number of Observations Used'])

    for key, value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print (dfoutput)
test_stationarity(df = df2, ts = 'ts')
def plot_transformed_data(df, ts, ts_transform):

  """

  Plot transformed and original time series data

  """

  f, ax = plt.subplots(1,1)

  ax.plot(df[ts])

  ax.plot(df[ts_transform], color = 'red')

  ax.set_title('%s and %s time-series graph' %(ts, ts_transform))

  ax.tick_params(axis = 'x', rotation = 45)

  ax.legend([ts, ts_transform])

  plt.show()

  plt.close()

  return
# Transformation - log ts

df2['ts_log'] = df2['ts'].apply(lambda x: np.log(x))



# Transformation - 7-day moving averages of log ts

df2['ts_log_moving_avg'] = df2['ts_log'].rolling(window = 7,

                                                               center = False).mean()



# Transformation - 7-day moving average ts

df2['ts_moving_avg'] = df2['ts'].rolling(window = 7,

                                                       center = False).mean()



# Transformation - Difference between logged ts and first-order difference logged ts

# df2['ts_log_diff'] = df2['ts_log'] - df2['ts_log'].shift()

df2['ts_log_diff'] = df2['ts_log'].diff()



# Transformation - Difference between ts and moving average ts

df2['ts_moving_avg_diff'] = df2['ts'] - df2['ts_moving_avg']



# Transformation - Difference between logged ts and logged moving average ts

df2['ts_log_moving_avg_diff'] = df2['ts_log'] - df2['ts_log_moving_avg']



# Transformation - Difference between logged ts and logged moving average ts

df2_transform = df2.dropna()



# Transformation - Logged exponentially weighted moving averages (EWMA) ts

df2_transform['ts_log_ewma'] = df2_transform['ts_log'].ewm(halflife = 7,

                                                                         ignore_na = False,

                                                                         min_periods = 0,

                                                                         adjust = True).mean()



# Transformation - Difference between logged ts and logged EWMA ts

df2_transform['ts_log_ewma_diff'] = df2_transform['ts_log'] - df2_transform['ts_log_ewma']



# Display data

display(df2_transform.head())



# Plot data

plot_transformed_data(df = df2, 

                      ts = 'ts', 

                      ts_transform = 'ts_log')

# Plot data

plot_transformed_data(df = df2, 

                      ts = 'ts_log', 

                      ts_transform = 'ts_log_moving_avg')



# Plot data

plot_transformed_data(df = df2_transform, 

                      ts = 'ts', 

                      ts_transform = 'ts_moving_avg')



# Plot data

plot_transformed_data(df = df2_transform, 

                      ts = 'ts_log', 

                      ts_transform = 'ts_log_diff')



# Plot data

plot_transformed_data(df = df2_transform, 

                      ts = 'ts', 

                      ts_transform = 'ts_moving_avg_diff')



# Plot data

plot_transformed_data(df = df2_transform, 

                      ts = 'ts_log', 

                      ts_transform = 'ts_log_moving_avg_diff')



# Plot data

plot_transformed_data(df = df2_transform, 

                      ts = 'ts_log', 

                      ts_transform = 'ts_log_ewma')



# Plot data

plot_transformed_data(df = df2_transform, 

                      ts = 'ts_log', 

                      ts_transform = 'ts_log_ewma_diff')



# Perform stationarity test

test_stationarity(df = df2_transform, 

                  ts = 'ts_log')



# Perform stationarity test

test_stationarity(df = df2_transform, 

                  ts = 'ts_moving_avg')



# Perform stationarity test

test_stationarity(df = df2_transform, 

                  ts = 'ts_log_moving_avg')



# Perform stationarity test

test_stationarity(df = df2_transform,

                  ts = 'ts_log_diff')



# Perform stationarity test

test_stationarity(df = df2_transform,

                  ts = 'ts_moving_avg_diff')



# Perform stationarity test

test_stationarity(df = df2_transform,

                  ts = 'ts_log_moving_avg_diff')



# Perform stationarity test

test_stationarity(df = df2_transform, 

                  ts = 'ts_log_ewma')



# Perform stationarity test

test_stationarity(df = df2_transform,

                  ts = 'ts_log_ewma_diff')
def plot_decomposition(df, ts, trend, seasonal, residual):

  """

  Plot time series data

  """

  f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize = (15, 5), sharex = True)



  ax1.plot(df[ts], label = 'Original')

  ax1.legend(loc = 'best')

  ax1.tick_params(axis = 'x', rotation = 45)



  ax2.plot(df[trend], label = 'Trend')

  ax2.legend(loc = 'best')

  ax2.tick_params(axis = 'x', rotation = 45)



  ax3.plot(df[seasonal],label = 'Seasonality')

  ax3.legend(loc = 'best')

  ax3.tick_params(axis = 'x', rotation = 45)



  ax4.plot(df[residual], label = 'Residuals')

  ax4.legend(loc = 'best')

  ax4.tick_params(axis = 'x', rotation = 45)

  plt.tight_layout()



  # Show graph

  plt.suptitle('Trend, Seasonal, and Residual Decomposition of %s' %(ts), 

               x = 0.5, 

               y = 1.05, 

               fontsize = 18)

  plt.show()

  plt.close()

  

  return
df_example_transform = df2_transform

from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(df_example_transform['ts_log'], freq = 365)



df_example_transform.loc[:,'trend'] = decomposition.trend

df_example_transform.loc[:,'seasonal'] = decomposition.seasonal

df_example_transform.loc[:,'residual'] = decomposition.resid



plot_decomposition(df = df_example_transform, 

                   ts = 'ts_log', 

                   trend = 'trend',

                   seasonal = 'seasonal', 

                   residual = 'residual')



test_stationarity(df = df_example_transform.dropna(), ts = 'residual')
#ARIMA Model

def plot_acf_pacf(df, ts):

  """

  Plot auto-correlation function (ACF) and partial auto-correlation (PACF) plots

  """

  f, (ax1, ax2) = plt.subplots(1,2, figsize = (10, 5)) 



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
#ACF and PACF plots:

from statsmodels.tsa.stattools import acf, pacf



# determine ACF and PACF

lag_acf = acf(np.array(df_example_transform['ts_log_diff']), nlags = 20)

lag_pacf = pacf(np.array(df_example_transform['ts_log_diff']), nlags = 20)



# plot ACF and PACF

plot_acf_pacf(df = df_example_transform, ts = 'ts_log_diff')
def run_arima_model(df, ts, p, d, q):

  """

  Run ARIMA model

  """

  from statsmodels.tsa.arima_model import ARIMA



  # fit ARIMA model on time series

  model = ARIMA(df[ts], order=(p, d, q))  

  results_ = model.fit(disp=-1)  

  

  # get lengths correct to calculate RSS

  len_results = len(results_.fittedvalues)

  ts_modified = df[ts][-len_results:]

  

  # calculate root mean square error (RMSE) and residual sum of squares (RSS)

  rss = sum((results_.fittedvalues - ts_modified)**2)

  rmse = np.sqrt(rss / len(df[ts]))

  

  # plot fit

  plt.plot(df[ts])

  plt.plot(results_.fittedvalues, color = 'red')

  plt.title('For ARIMA model (%i, %i, %i) for ts %s, RSS: %.4f, RMSE: %.4f' %(p, d, q, ts, rss, rmse))

  

  plt.show()

  plt.close()

  

  return results_
# Note: I do the differencing in the transformation of the data 'ts_log_diff'

# AR model with 1st order differencing - ARIMA (1,0,0)

model_AR = run_arima_model(df = df_example_transform, 

                           ts = 'ts_log_diff', 

                           p = 1, 

                           d = 0, 

                           q = 0)



# MA model with 1st order differencing - ARIMA (0,0,1)

model_MA = run_arima_model(df = df_example_transform, 

                           ts = 'ts_log_diff', 

                           p = 0, 

                           d = 0, 

                           q = 1)



# ARMA model with 1st order differencing - ARIMA (0,1,0)

model_MA = run_arima_model(df = df_example_transform, 

                           ts = 'ts_log_diff', 

                           p = 0, 

                           d = 1, 

                           q = 1)
from fbprophet import Prophet

import datetime

from datetime import datetime
df_example = df2

def days_between(d1, d2):

    """Calculate the number of days between two dates.  D1 is start date (inclusive) and d2 is end date (inclusive)"""

    d1 = datetime.strptime(d1, "%Y-%m-%d")

    d2 = datetime.strptime(d2, "%Y-%m-%d")

    return abs((d2 - d1).days + 1)



# Inputs for query



date_column = 'Date'

metric_column = 'ts'

table = df_example

start_training_date = '2004-08-19'

end_training_date = '2019-08-12'

start_forecasting_date = '2019-08-13'

end_forecasting_date = '2019-12-31'

year_to_estimate = '2019'



# Inputs for forecasting



# future_num_points

# If doing different time intervals, change future_num_points

future_num_points = days_between(start_forecasting_date, end_forecasting_date)



cap = None # 2e6



# growth: default = 'linear'

# Can also choose 'logistic'

growth = 'linear'



# n_changepoints: default = 25, uniformly placed in first 80% of time series

n_changepoints = 25 



# changepoint_prior_scale: default = 0.05

# Increasing it will make the trend more flexible

changepoint_prior_scale = 0.05 



# changpoints: example = ['2016-01-01']

changepoints = None 



# holidays_prior_scale: default = 10

# If you find that the holidays are overfitting, you can adjust their prior scale to smooth them

holidays_prior_scale = 10 



# interval_width: default = 0.8

interval_width = 0.8 



# mcmc_samples: default = 0

# By default Prophet will only return uncertainty in the trend and observation noise.

# To get uncertainty in seasonality, you must do full Bayesian sampling. 

# Replaces typical MAP estimation with MCMC sampling, and takes MUCH LONGER - e.g., 10 minutes instead of 10 seconds.

# If you do full sampling, then you will see the uncertainty in seasonal components when you plot:

mcmc_samples = 0



holidays = None



daily_seasonality = True
# get relevant data - note: could also try this with ts_log_diff

df_prophet = df_example_transform[['ts']] # can try with ts_log_diff



# reset index

df_prophet = df_prophet.reset_index()



# rename columns

df_prophet = df_prophet.rename(columns = {'Date': 'ds', 'ts': 'y'}) # can try with ts_log_diff



# Change 'ds' type from datetime to date (necessary for FB Prophet)

df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])



# Change 'y' type to numeric (necessary for FB Prophet)

df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='ignore')



# Remove any outliers

# df.loc[(df_['ds'] > '2016-12-13') & (df_['ds'] < '2016-12-19'), 'y'] = None
def create_daily_forecast(df,

#                           cap,

                          holidays,

                          growth,

                          n_changepoints = 25,

                          changepoint_prior_scale = 0.05,

                          changepoints = None,

                          holidays_prior_scale = 10,

                          interval_width = 0.8,

                          mcmc_samples = 1,

                          future_num_points = 10, 

                          daily_seasonality = True):

  """

  Create forecast

  """

  

  # Create copy of dataframe

  df_ = df.copy()



  # Add in growth parameter, which can change over time

  #     df_['cap'] = max(df_['y']) if cap is None else cap



  # Create model object and fit to dataframe

  m = Prophet(growth = growth,

              n_changepoints = n_changepoints,

              changepoint_prior_scale = changepoint_prior_scale,

              changepoints = changepoints,

              holidays = holidays,

              holidays_prior_scale = holidays_prior_scale,

              interval_width = interval_width,

              mcmc_samples = mcmc_samples, 

              daily_seasonality = daily_seasonality)



  # Fit model with dataframe

  m.fit(df_)



  # Create dataframe for predictions

  future = m.make_future_dataframe(periods = future_num_points)

  #     future['cap'] = max(df_['y']) if cap is None else cap



  # Create predictions

  fcst = m.predict(future)



  # Plot

  m.plot(fcst);

  m.plot_components(fcst)



  return fcst
fcst = create_daily_forecast(df_prophet,

#                              cap,

                             holidays,

                             growth,

                             n_changepoints,

                             changepoint_prior_scale,

                             changepoints, 

                             holidays_prior_scale,

                             interval_width,

                             mcmc_samples,

                             future_num_points, 

                             daily_seasonality)
def calculate_mape(y_true, y_pred):

    """ Calculate mean absolute percentage error (MAPE)"""

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



def calculate_mpe(y_true, y_pred):

    """ Calculate mean percentage error (MPE)"""

    return np.mean((y_true - y_pred) / y_true) * 100



def calculate_mae(y_true, y_pred):

    """ Calculate mean absolute error (MAE)"""

    return np.mean(np.abs(y_true - y_pred)) * 100



def calculate_rmse(y_true, y_pred):

    """ Calculate root mean square error (RMSE)"""

    return np.sqrt(np.mean((y_true - y_pred)**2))



def print_error_metrics(y_true, y_pred):

    print('MAPE: %f'%calculate_mape(y_true, y_pred))

    print('MPE: %f'%calculate_mpe(y_true, y_pred))

    print('MAE: %f'%calculate_mae(y_true, y_pred))

    print('RMSE: %f'%calculate_rmse(y_true, y_pred))

    return
print_error_metrics(y_true = df_prophet['y'], y_pred = fcst['yhat'])
#RNN 



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

do_lstm_model(df = df_prophet, 

              ts = 'y', 

              look_back = 1, 

              epochs = 5)



# LSTM for Regression Using the Window Method

do_lstm_model(df = df_prophet, 

              ts = 'y', 

              look_back = 3, 

              epochs = 5)



# LSTM for Regression with Time Steps

do_lstm_model(df = df_prophet, 

              ts = 'y', 

              look_back = 3, 

              epochs = 5, 

              type_ = 'regression with time steps')