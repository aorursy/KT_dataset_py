import os
# import packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from IPython.display import display, HTML, display_html

import seaborn as sns

import datetime



# set formatting

pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 100)



# read in CSV file data

df = pd.read_csv('../input/reviews.csv')



print("Setup Complete")
# look at data

display(df.head())
# shape of data

display(df.shape)
# look at data types

display(df.dtypes)
# see if there are any null values

display(df.isnull().any())
# display descriptive statistics

display(df.describe(percentiles = [0.25, 0.5, 0.75, 0.85, 0.95, 0.99]))
# Rename columns

df = df.rename(columns = {'date': 'ds', 'listing_id': 'ts'})



# Group data

df_group = df.groupby(by = 'ds').agg({'ts':'count'})



# change index to datetime

df_group.index = pd.to_datetime(df_group.index)



# Set frequncy of time series

df_group = df_group.asfreq(freq = '1D')



# Sort the values

df_group = df_group.sort_index(ascending = True)



# Fill NA values with zero

df_group = df_group.fillna(value = 0)



# Show the end of th data

display(df_group.tail())
# Plot time series data

f, ax = plt.subplots(1,1)

ax.plot(df_group['ts'])



# Add title

ax.set_title('Time-series Graph')



# Rotate x-labels

ax.tick_params(axis= 'x', rotation = 45)



# show graph

plt.show()

plt.close()
from statsmodels.tsa.stattools import adfuller

def test_stationarity(df, ts):

    """

    Test Stationarity using moving average statistics and Dickey-Fuller Test

    """

    

    # Determining rolling statistics

    rolmean = df[ts].rolling(window = 12, center = False).mean()

    rolstd = df[ts].rolling(window = 12, center = False).std()

    

    # Plot rolling statistics:

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

    

    print('Dickey-Fuller Test:')

    print('Null Hypothesis (H_0): time series is not stationary')

    print('ALternative Hypothesis (H_1): time series is stationary')

    print ('Results of Dickey-Fuller Test:')

    dftest = adfuller(df[ts],

                     autolag = 'AIC')

    dfoutput = pd.Series(dftest[0:4],

                        index = ['Test Statistics',

                                'p-value',

                                '# Lags Used',

                                'Number of observation Used'])

    for key, value in dftest[4].items():

        dfoutput['Critical Value (%s)' %key] = value

        

    if (dftest[1] <= 0.05):

        print('Reject the null hypothesis (H0), the data does not have a unit root and is stationary.')

    else:

        print('Fail to reject the null hypothesis (H0), the data has a unit root and is non-stationary.')

    

    print(dfoutput)
test_stationarity(df = df_group, ts = 'ts')
def plot_transformed_data(df, ts, ts_transform):

    """

    Plot transform and original time series data

    """

    # Plot time series data

    f, ax = plt.subplots(1,1)

    ax.plot(df[ts])

    ax.plot(df[ts_transform], color = 'red')

    

    # Add title

    ax.set_title('%s and %s time-series graph' %(ts, ts_transform))

    

    # Rotate x-labels

    ax.tick_params(axis = 'x', rotation = 45)

    

    # Add legend

    ax.legend([ts, ts_transform])

    

    plt.show()

    plt.close()

    

    return
# Transformation - log ts

df_group['ts_log'] = df_group['ts'].apply(lambda x: np.log(x))
# Transformation - 7-day moving average of log ts

df_group['ts_log_moving_avg'] = df_group['ts_log'].rolling(window = 7,

                                                              center = False).mean()
# Transformation - 7-day moving average of ts

df_group['ts_moving_avg'] = df_group['ts'].rolling(window = 7, 

                                                  center = False).mean()
# Transformation - Difference betwen logged ts and first-order difference logged ts

# df_group['ts_log_diff'] = df_group['ts_log'] - df_group['ts_log'].shift()

df_group['ts_log_diff'] = df_group['ts_log'].diff()
# Transformation - Differencing between ts and moving average ts

df_group['ts_moving_avg_diff'] = df_group['ts'] - df_group['ts_moving_avg']
# Transformation - Difference between logged ts and logged moving average ts

df_group['ts_log_moving_avg_diff'] = df_group['ts_log'] - df_group['ts_log_moving_avg']
df_group_transform = df_group.dropna()
# Transformation - Logged exponentailly weighted moving averages (EWMA) ts

df_group_transform['ts_log_ewma'] = df_group_transform['ts_log'].ewm(halflife = 7,

                                                                    ignore_na = False,

                                                                    min_periods = 0,

                                                                    adjust = True).mean()
# Transformation - Difference between logged ts and logged EWMA ts

df_group_transform['ts_log_ewma_diff'] = df_group_transform['ts_log'] - df_group_transform['ts_log_ewma']
display(df_group_transform.head())
# Plot Data

plot_transformed_data(df = df_group,

                     ts = 'ts',

                     ts_transform = 'ts_log')



plot_transformed_data(df = df_group,

                     ts = 'ts_log',

                     ts_transform = 'ts_log_moving_avg')



plot_transformed_data(df = df_group_transform,

                     ts = 'ts',

                     ts_transform = 'ts_moving_avg')



plot_transformed_data(df = df_group_transform,

                     ts = 'ts_log',

                     ts_transform = 'ts_log_diff')



plot_transformed_data(df = df_group_transform,

                   ts = 'ts',

                   ts_transform = 'ts_moving_avg_diff')



plot_transformed_data(df = df_group_transform,

                   ts = 'ts_log',

                   ts_transform = 'ts_log_moving_avg_diff')



plot_transformed_data(df = df_group_transform,

                   ts = 'ts_log',

                   ts_transform = 'ts_log_ewma')



plot_transformed_data(df = df_group_transform,

                   ts = 'ts_log',

                   ts_transform = 'ts_log_ewma_diff')

# Perform Stationarity Test

test_stationarity(df = df_group_transform,

                 ts = 'ts_log')

test_stationarity(df = df_group_transform,

                 ts = 'ts_moving_avg')

test_stationarity(df = df_group_transform,

                 ts = 'ts_log_moving_avg')

test_stationarity(df = df_group_transform,

                 ts = 'ts_log_diff')

test_stationarity(df = df_group_transform,

                 ts = 'ts_moving_avg_diff')

test_stationarity(df = df_group_transform,

                 ts = 'ts_log_moving_avg_diff')

test_stationarity(df = df_group_transform,

                 ts = 'ts_log_ewma')

test_stationarity(df = df_group_transform,

                 ts = 'ts_log_ewma_diff')
def plot_decomposition(df, ts, trend, seasonal, residual):

    """

    Plot Time seris data

    """

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize = (15, 5), sharex = True)

    

    ax1.plot(df[ts], label = 'Original')

    ax1.legend(loc = 'best')

    ax1.tick_params(axis = 'x', rotation = 45)

    

    ax2.plot(df[trend], label = 'Trend')

    ax2.legend(loc = 'best')

    ax2.tick_params(axis = 'x', rotation = 45)

    

    ax3.plot(df[seasonal], label = 'Seasonality')

    ax3.legend(loc = 'best')

    ax3.tick_params(axis = 'x', rotation = 45)

    

    ax4.plot(df[residual], label = 'Residuals')

    ax4.legend(loc = 'best')

    ax4.tick_params(axis = 'x', rotation = 45)

    

    # Show Graph

    plt.suptitle('Trend, Seasonal, and Residual Dcomposition of %s' %(ts),

                x = 0.5,

                y = 1.05,

                fontsize = 20)

    plt.show()

    plt.close()

    

    return
from statsmodels.tsa.seasonal import seasonal_decompose



decomposition = seasonal_decompose(df_group_transform['ts_log'], period = 365)



df_group_transform.loc[:, 'trend'] = decomposition.trend

df_group_transform.loc[:, 'seasonal'] = decomposition.seasonal

df_group_transform.loc[:, 'residual'] = decomposition.resid



plot_decomposition(df = df_group_transform,

                  ts = 'ts_log',

                  trend = 'trend',

                  seasonal = 'seasonal',

                  residual = 'residual')



test_stationarity(df = df_group_transform.dropna(), ts = 'residual')
def plot_acf_pacf(df,ts):

    """

    Plot auto-correlation (ACF) and partial auto-crrelation (PACF) plots

    """

    f, (ax1, ax2) = plt.subplots(1,2, figsize = (10, 5))

    

    # Plot ACF:

    

    ax1.plot(lag_acf)

    ax1.axhline(y = 0, linestyle = '--', color = 'gray')

    ax1.axhline(y = -1.96/np.sqrt(len(df[ts])), linestyle = '--', color = 'gray')

    ax1.axhline(y = 1.96/np.sqrt(len(df[ts])), linestyle = '--', color = 'gray')

    ax1.set_title('Autocorrelation Function for %s' %(ts))

    

    # Plot PACF:

    

    ax2.plot(lag_pacf)

    ax2.axhline(y = 0, linestyle = '--', color = 'gray')

    ax2.axhline(y = -1.96/np.sqrt(len(df[ts])), linestyle = '--', color = 'gray')

    ax2.axhline(y = 1.96/np.sqrt(len(df[ts])), linestyle = '--', color = 'gray')

    ax2.set_title('Partial Autocorrelation Function for %s' %(ts))

    

    plt.tight_layout()

    plt.show()

    plt.close()

    

    return
# ACF and PACF plots:

from statsmodels.tsa.stattools import acf, pacf



# determine ACF and PACF

lag_acf = acf(np.array(df_group_transform['ts_log_diff']), nlags = 20, fft = False)

lag_pacf = pacf(np.array(df_group_transform['ts_log_diff']), nlags = 20)



# plot ACF and PACF

plot_acf_pacf(df = df_group_transform, ts = 'ts_log_diff')
def run_arima_model(df, ts, p, d, q):

    """

    Run ARIMA model

    """

    from statsmodels.tsa.arima_model import ARIMA

    

    # fit ARIMA model on time series

    model = ARIMA(df[ts], order = (p, d, q))

    results = model.fit(disp = -1)

    

    # get lenghts correct to calculate RSS

    len_results = len(results.fittedvalues)

    ts_modified = df[ts][-len_results:]

    

    # calculate root mean square error (RMSE) and residual sun of Squares (RSS)

    rss = sum((results.fittedvalues - ts_modified)**2)

    rmse = np.sqrt(rss / len(df[ts]))

    

    # Plot fit 

    plt.plot(df[ts])

    plt.plot(results.fittedvalues, color = 'red')

    plt.title('ARIMA model (%i, %i, %i) for ts %s, RSS:  %.4f, RMSE: %.4f' %(p, d, q, ts, rss, rmse))

    

    plt.show()

    plt.close()

    

    return results
# Note: We have already done differencing in the transformation of data 'ts_log_diff'

# AR model with 1st order differencing - ARIMA (1, 0, 0)

model_AR = run_arima_model(df = df_group_transform,

                          ts = 'ts_log_diff',

                          p = 1,

                          d = 0,

                          q = 0)



# MA model with 1st order differencing - ARIMA (0, 0, 1)

model_MA = run_arima_model(df = df_group_transform,

                          ts = 'ts_log_diff',

                          p = 0,

                          d = 0,

                          q = 1)



# ARIMA model with 1st order differencing - ARIMA (1, 0, 1)

model_ARIMA = run_arima_model(df = df_group_transform,

                          ts = 'ts_log_diff',

                          p = 1,

                          d = 0,

                          q = 1)
from fbprophet import Prophet

import datetime

from datetime import datetime
def days_between(d1, d2):

    """Calculate the number of days between two dates.  D1 is start date (inclusive) and d2 is end date (inclusive)"""

    d1 = datetime.strptime(d1, "%Y-%m-%d")

    d2 = datetime.strptime(d2, "%Y-%m-%d")

    return abs((d2 - d1).days + 1)
df_group
# Inputs for query



date_column = 'dt'

metric_column = 'ts'

table = df_group

start_training_date = '2010-07-03'

end_training_date = '2020-05-08'

start_forecasting_date = '2020-05-09'

end_forecasting_date = '2020-12-31'

year_to_estimate = '2020'



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



# holiday: default = None

# thanksgiving = pd.DataFrame({

#   'holiday': 'thanksgiving',

#   'ds': pd.to_datetime(['2014-11-27', '2015-11-26',

#                         '2016-11-24', '2017-11-23']),

#   'lower_window': 0,

#   'upper_window': 4,

# })

# christmas = pd.DataFrame({

#   'holiday': 'christmas',

#   'ds': pd.to_datetime(['2014-12-25', '2015-12-25', 

#                         '2016-12-25','2017-12-25']),

#   'lower_window': -1,

#   'upper_window': 0,

# })

# holidays = pd.concat((thanksgiving,christmas))

holidays = None



daily_seasonality = True
df_group_transform
df_prophet = df_group_transform[['ts']]
df_prophet
# get relevant data - note: could also try this with ts_log_diff

df_prophet = df_group_transform[['ts']] # can try with ts_log_diff



# reset index

df_prophet = df_prophet.reset_index()



# rename columns

df_prophet = df_prophet.rename(columns = {'ds': 'ds', 'ts': 'y'}) # can try with ts_log_diff



# Change 'ds' type from datetime to date (necessary for FB Prophet)

df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])



# Change 'y' type to numeric (necessary for FB Prophet)

df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='ignore')



# Remove any outliers

# df.loc[(df_['ds'] > '2016-12-13') & (df_['ds'] < '2016-12-19'), 'y'] = None
df_prophet
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


def do_lstm_model(df, 

                  ts, 

                  look_back, 

                  epochs, 

                  type_ = None, 

                  train_fraction = 0.67):

  """

   Create LSTM model

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