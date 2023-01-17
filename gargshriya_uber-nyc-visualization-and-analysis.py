import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import cm

import matplotlib.pyplot as plt

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

import statsmodels.api as sm



from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import r2_score



def adj_r2_score(r2, n, k):

    return 1-((1-r2)*((n-1)/(n-k-1)))

from sklearn.metrics import mean_squared_error

from math import sqrt



from keras.models import Sequential

from keras.layers import Dense

import keras.backend as K

from keras.callbacks import EarlyStopping

from keras.optimizers import Adam

from keras.models import load_model
uber_raw_apr14 = pd.read_csv("../input/uber-raw-data-apr14.csv")

uber_raw_may14 = pd.read_csv("../input/uber-raw-data-may14.csv")

uber_raw_jun14 = pd.read_csv("../input/uber-raw-data-jun14.csv")

uber_raw_jul14 = pd.read_csv("../input/uber-raw-data-jul14.csv")

uber_raw_aug14 = pd.read_csv("../input/uber-raw-data-aug14.csv")

uber_raw_sep14 = pd.read_csv("../input/uber-raw-data-sep14.csv")





#Combining dataset of 6 months into 1 dataset

uber_2014 = [uber_raw_apr14, uber_raw_may14, uber_raw_jun14, uber_raw_jul14,uber_raw_aug14, uber_raw_sep14]

uber_data_2014 = pd.concat(uber_2014,axis=0,ignore_index=True)

uber_data_2014.head()
uber_data_2014.info()
uber_data_2014.Timestamp = pd.to_datetime(uber_data_2014['Date/Time'],format='%m/%d/%Y %H:%M:%S') 

uber_data_2014['Date_only'] = uber_data_2014.Timestamp.dt.date

uber_data_2014['Date'] = uber_data_2014.Timestamp

uber_data_2014['Month'] = uber_data_2014.Timestamp.dt.month

uber_data_2014['DayOfWeekNum'] = uber_data_2014.Timestamp.dt.dayofweek

uber_data_2014['DayOfWeek'] = uber_data_2014.Timestamp.dt.weekday_name

uber_data_2014['MonthDayNum'] = uber_data_2014.Timestamp.dt.day

uber_data_2014['HourOfDay'] = uber_data_2014.Timestamp.dt.hour



uber_data_2014= uber_data_2014.drop(columns = ['Lat','Lon'])

uber_data_2014.tail()
uber_data_2014.groupby(pd.Grouper(key='DayOfWeek')).count()



uber_weekdays = uber_data_2014.pivot_table(index=['DayOfWeekNum','DayOfWeek'],

                                  values='Base',

                                  aggfunc='count')

uber_weekdays.plot(kind='bar', figsize=(15,8))

plt.ylabel('Total Journeys')

plt.xlabel('Day')

plt.title('Journeys by Week Day');
uber_hour = uber_data_2014.pivot_table(index=['HourOfDay'],

                                  values='Base',

                                  aggfunc='count')

uber_hour.plot(kind='bar', figsize=(8,6))

plt.ylabel('Total Journeys')

plt.title('Journeys by Hour');
uber_data_2014.groupby(pd.Grouper(key='Base')).count()



uber_monthdays = uber_data_2014.pivot_table(index=['Base'], values='Date' ,

                                  aggfunc='count')

uber_monthdays.plot(kind='bar', figsize=(8,6))

plt.ylabel('Total Journeys')

plt.title('Journeys by Month Day');
uber_data_2014= uber_data_2014.drop(columns = ['Month','DayOfWeekNum','Base', 'DayOfWeek', 'MonthDayNum', 'HourOfDay'])

#uber_data_2014.tail()
'''

The df uber_count is the grouping of the above dataset on hourly basis with time stamp of both date and time.

This df is used mostly for ANN analysis.

'''

uber_count=uber_data_2014.groupby(pd.Grouper(key='Date')).count()

uber_count= uber_count.drop(columns = ['Date_only'])

print(uber_count.info())



train = uber_count[:][:234083]             #90% of 260093

test = uber_count[:][234084:]

display(train.tail())

test.head()

train['Date/Time'].plot(kind='line',figsize=(15,8), title= 'Hourly Ridership', fontsize=14)

test['Date/Time'].plot(figsize=(15,5), title= 'Hourly Ridership', fontsize=14)

plt.ylabel('Total Journeys')

plt.xlabel('Month')

plt.show()
'''

The df uber_dates is the grouping of the above dataset on daily basis with time stamp of onlu date.

This df is used to for univariate Time Series Forecasting.

'''

uber_dates=uber_data_2014.groupby(pd.Grouper(key='Date_only')).count()

uber_dates= uber_dates.drop(columns = ['Date'])

print(uber_dates.info())

uber_dates_d= uber_dates.drop(columns = ['Date/Time'])



train_ts = uber_dates[:][:163]                     #split is 90-10

test_ts = uber_dates[:][164:]

test_ts_d = uber_dates_d[:][164:]

test_ts.head()
train_ts['Date/Time'].plot(kind='line',figsize=(15,8), title= 'Daily Ridership', fontsize=14)

test_ts['Date/Time'].plot(figsize=(15,5), title= 'Daily Ridership', fontsize=14)

plt.ylabel('Total Journeys')

plt.xlabel('Month')

plt.show()
y_hat_avg = test_ts.copy()

fit1 = ExponentialSmoothing(np.asarray(train_ts['Date/Time'].astype(float)) ,seasonal_periods=7 ,trend='add', seasonal='add',).fit()

y_hat_avg['Holt_Winter'] = fit1.forecast(len(test_ts))

plt.figure(figsize=(15,5))

plt.plot( train_ts['Date/Time'], label='Train')

plt.plot(test_ts['Date/Time'], label='Test')

plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')

plt.legend(loc='best')

plt.ylabel('Total Journeys')

plt.xlabel('Months')

plt.show()
rmse = sqrt(mean_squared_error(test_ts['Date/Time'], y_hat_avg['Holt_Winter']))

rmse
y_hat_avg = test_ts.copy()

fit1 = sm.tsa.statespace.SARIMAX(train_ts['Date/Time'], order=(2, 1, 4),seasonal_order=(1,1,1,7)).fit()

y_hat_avg['SARIMA'] = fit1.predict(start="2014-09-11", end="2014-09-30", dynamic=True)

plt.figure(figsize=(15,6))

plt.plot( train_ts['Date/Time'], label='Train')

plt.plot(test_ts['Date/Time'], label='Test')

plt.plot(y_hat_avg['SARIMA'], label='SARIMA')

plt.legend(loc='best')

plt.ylabel('Total Journeys')

plt.xlabel('Months')

plt.show()

rms = sqrt(mean_squared_error(test_ts['Date/Time'], y_hat_avg.SARIMA))

print(rms)     
plt.style.use('default')

plt.figure(figsize = (16,8))

import statsmodels.api as sm

sm.tsa.seasonal_decompose(train_ts['Date/Time'].values,freq=30).plot()

result = sm.tsa.stattools.adfuller(uber_dates['Date/Time'])

plt.show()
y_hat_avg = test_ts.copy()



fit1 = Holt(np.asarray(train_ts['Date/Time']).astype(float)).fit(smoothing_level = 0.3,smoothing_slope = 0.1)

y_hat_avg['Holt_linear'] = fit1.forecast(len(test_ts))



plt.figure(figsize=(16,5))

plt.plot(train_ts['Date/Time'], label='Train')

plt.plot(test_ts['Date/Time'], label='Test')

plt.plot(y_hat_avg['Holt_linear'], label='Holt_linear')

plt.legend(loc='best')

plt.show()
rms = sqrt(mean_squared_error(test_ts['Date/Time'], y_hat_avg.Holt_linear))

print(rms)     
from statsmodels.tsa.stattools import adfuller

def test_stationary(timeseries):

    #Determine rolling statistics

    #rolmean = pd.rolling_mean(timeseries,window = 24)

    #rolstd = pd.rolling_std(timeseries, window = 24)

    

    rolmean = timeseries.rolling(24).mean()

    rolstd = timeseries.rolling(24).std()

    

    

    #Plot rolling Statistics

    orig = plt.plot(timeseries, color = "blue", label = "Original")

    mean = plt.plot(rolmean, color = "red", label = "Rolling Mean")

    std = plt.plot(rolstd, color = "black", label = "Rolling Std")

    plt.legend(loc = "best")

    plt.title("Rolling Mean and Standard Deviation")

    plt.show(block = False)

    

    #Perform Dickey Fuller test

    print("Results of Dickey Fuller test: ")

    dftest = adfuller(timeseries, autolag = 'AIC')

    dfoutput = pd.Series(dftest[0:4], index = ['Test Statistics', 'p-value', '# Lag Used', 'Number of Observations Used'])

    

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)' %key] = value

    print(dfoutput)
from matplotlib.pylab import rcParams

rcParams['figure.figsize']=(20,10)

test_stationary(uber_count['Date/Time'])
Train_log = np.log(train_ts['Date/Time'])

valid_log = np.log(test_ts['Date/Time'])
moving_avg = Train_log.rolling(24).mean()

plt.plot(Train_log)

plt.plot(moving_avg, color = 'red')
train_log_moving_diff = Train_log - moving_avg

train_log_moving_diff.dropna(inplace = True)

test_stationary(train_log_moving_diff)
train_log_diff = Train_log - Train_log.shift(1)

test_stationary(train_log_diff.dropna())
from statsmodels.tsa.seasonal import seasonal_decompose

plt.figure(figsize = (16,10))

decomposition = seasonal_decompose(pd.DataFrame(Train_log)['Date/Time'].values, freq = 24)

plt.style.use('default')

trend = decomposition.trend

seasonal = decomposition.seasonal

residual = decomposition.resid



plt.subplot(411)

plt.plot(Train_log, label = 'Original')

plt.legend(loc = 'best')

plt.subplot(412)

plt.plot(trend, label = 'Trend')

plt.legend(loc = 'best')

plt.subplot(413)

plt.plot(seasonal, label = 'Seasonal')

plt.legend(loc = 'best')

plt.subplot(414)

plt.plot(residual, label = 'Residuals')

plt.legend(loc = 'best')

plt.tight_layout()
plt.figure(figsize = (16,8))

train_log_decompose = pd.DataFrame(residual)

train_log_decompose['date'] = Train_log.index

train_log_decompose.set_index('date', inplace = True)

train_log_decompose.dropna(inplace = True)

test_stationary(train_log_decompose[0])
from statsmodels.tsa.stattools import acf, pacf



lag_acf = acf(train_log_diff.dropna(), nlags = 25)

lag_pacf = pacf(train_log_diff.dropna(), nlags = 25, method= "ols")
plt.figure(figsize = (15,8))

plt.style.use("fivethirtyeight")

plt.plot(lag_acf)

plt.axhline( y = 0, linestyle = "--", color = "gray")

plt.axhline( y= -1.96/np.sqrt(len(train_log_diff.dropna())), linestyle = "--", color = "gray")

plt.axhline(y = 1.96 /np.sqrt(len(train_log_diff.dropna())), linestyle = "--", color = "gray")

plt.title("Autocorrelation Function")

plt.show()

# PACF

plt.figure(figsize = (15,8))

plt.plot(lag_pacf)

plt.axhline(y = 0, linestyle = "--", color = "gray")

plt.axhline(y = -1.96/np.sqrt(len(train_log_diff.dropna())), linestyle = "--", color = "gray")

plt.axhline( y = 1.96/np.sqrt(len(train_log_diff.dropna())), linestyle = "--", color = "gray")

plt.title("Partial Autocorrelation Function")

plt.show()
from statsmodels.tsa.arima_model import ARIMA

plt.figure(figsize = (15,8))

model = ARIMA(Train_log, order = (2,1,0))  #here q value is zero since it is just AR Model

results_AR = model.fit(disp=-1)

plt.plot(train_log_diff.dropna(), label = "Original")

plt.plot(results_AR.fittedvalues, color = 'red', label = 'Predictions')

plt.legend(loc = 'best')
AR_predict = results_AR.predict(start="2014-09-11", end="2014-09-30")

AR_predict = AR_predict.cumsum().shift().fillna(0)

AR_predict1 = pd.Series(np.ones(test_ts.shape[0])* np.log(test_ts['Date/Time'])[0], index = test_ts_d)

AR_predict = np.exp(AR_predict1)
# Moving Average Model
plt.figure(figsize = (15,8))

model = ARIMA(Train_log, order = (0,1,2)) # here the p value is 0 since it is moving average model

results_MA = model.fit(disp = -1)

plt.plot(train_log_diff.dropna(), label = "Original")

plt.plot(results_MA.fittedvalues, color = "red", label = "Prediction")

plt.legend(loc = "best")
MA_predict = results_MA.predict(start="2014-09-11", end="2014-09-30")

MA_predict=MA_predict.cumsum().shift().fillna(0)

MA_predict1=pd.Series(np.ones(test_ts.shape[0]) * np.log(test_ts['Date/Time'])[0], index = test_ts_d)

#MA_predict1=MA_predict1.add(MA_predict,fill_value=0)

MA_predict = np.exp(MA_predict1)
# Combined Model
plt.figure(figsize = (16,8))

model = ARIMA(Train_log, order=(2, 1, 2))  

results_ARIMA = model.fit(disp=-1)  

plt.plot(train_log_diff.dropna(),  label='Original')

plt.plot(results_ARIMA.fittedvalues, color='red', label='Predicted')

plt.legend(loc='best')

plt.show()
# Function to scale model to original scale
def check_prediction_diff(predict_diff, given_set):

    predict_diff= predict_diff.cumsum().shift().fillna(0)

    predict_base = pd.Series(np.ones(given_set.shape[0]) * np.log(given_set['Date/Time'])[0], index = given_set.index)

    #predict_log = predict_base.add(predict_diff,fill_value=0)

    predict = np.exp(predict_base)

    

    plt.plot(given_set['Date/Time'], label = "Given set")

    plt.plot(predict, color = 'red', label = "Predict")

    plt.legend(loc= 'best')

    plt.title('RMSE: %.4f'% (np.sqrt(np.dot(predict, given_set['Date/Time']))/given_set.shape[0]))

    plt.show()



def check_prediction_log(predict_log, given_set):

    predict = np.exp(predict_log)

    

    plt.plot(given_set['Date/Time'], label = "Given set")

    plt.plot(predict, color = 'red', label = "Predict")

    plt.legend(loc= 'best')

    plt.title('RMSE: %.4f'% (np.sqrt(np.dot(predict, given_set['Date/Time']))/given_set.shape[0]))

    plt.show()



ARIMA_predict_diff=results_ARIMA.predict(start="2014-09-11", end="2014-09-30")



plt.figure(figsize = (16,8))

check_prediction_diff(ARIMA_predict_diff, test_ts)
ARIMA_predict_diff.shape 





test_ts.shape
y_hat = test_ts.copy()

fit2 = SimpleExpSmoothing(np.asarray(train_ts['Date/Time'])).fit(smoothing_level = 0.6,optimized = False)

y_hat['SES'] = fit2.forecast(len(test_ts))

plt.figure(figsize =(15,8))

plt.plot(train_ts['Date/Time'], label = 'Train')

plt.plot(test_ts['Date/Time'], label = 'Validation')

plt.plot(y_hat['SES'], label = 'Simple Exponential Smoothing')

plt.legend(loc = 'best')
abc=y_hat['SES'].values.tolist()

rmse = sqrt(mean_squared_error(test_ts['Date/Time'],abc))

rmse
y_hat_avg = test_ts.copy()

y_hat_avg['moving_average_forecast'] = train_ts['Date/Time'].rolling(10).mean().iloc[-1]

plt.figure(figsize = (15,5))

plt.plot(train_ts['Date/Time'], label = 'Train')

plt.plot(test_ts['Date/Time'], label = 'Validation')

plt.plot(y_hat_avg['moving_average_forecast'], label = 'Moving Average Forecast with 10 Observations')

plt.legend(loc = 'best')

plt.show()

y_hat_avg = test_ts.copy()

y_hat_avg['moving_average_forecast'] = train_ts['Date/Time'].rolling(20).mean().iloc[-1]

plt.figure(figsize = (15,5))

plt.plot(train_ts['Date/Time'], label = 'Train')

plt.plot(test_ts['Date/Time'], label = 'Validation')

plt.plot(y_hat_avg['moving_average_forecast'],label = 'Moving Average Forecast with 20 Observations')

plt.legend(loc = 'best')

plt.show()

y_hat_avg = test_ts.copy()

y_hat_avg['moving_average_forecast']= train_ts['Date/Time'].rolling(50).mean().iloc[-1]

plt.figure(figsize = (15,5))

plt.plot(train_ts['Date/Time'], label = 'Train')

plt.plot(test_ts['Date/Time'], label = 'Validation')

plt.plot(y_hat_avg['moving_average_forecast'], label = "Moving Average Forecast with 50 Observations")

plt.legend(loc = 'best')

plt.show()
rmse = sqrt(mean_squared_error(test_ts['Date/Time'], y_hat_avg['moving_average_forecast']))

rmse
y_hat = test_ts.copy()

fit2 = SimpleExpSmoothing(np.asarray(train_ts['Date/Time'])).fit(smoothing_level = 0.6,optimized = False)

y_hat['SES'] = fit2.forecast(len(test_ts))

plt.figure(figsize =(15,8))

plt.plot(train_ts['Date/Time'], label = 'Train')

plt.plot(test_ts['Date/Time'], label = 'Validation')

plt.plot(y_hat['SES'], label = 'Simple Exponential Smoothing')

plt.legend(loc = 'best')

sc = MinMaxScaler()

train_sc = sc.fit_transform(train)

test_sc = sc.transform(test)



X_train = train_sc[:-1]

y_train = train_sc[1:]



X_test = test_sc[:-1]

y_test = test_sc[1:]
K.clear_session()



model = Sequential()

model.add(Dense(9, input_dim=1, activation='relu'))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)

history = model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=1, callbacks=[early_stop], shuffle=False)
y_pred_test_ann = model.predict(X_test)

y_train_pred_ann = model.predict(X_train)

rmse = sqrt(mean_squared_error(y_train,y_train_pred_ann))

print("Train : {:0.3f}".format(rmse))



rmse = sqrt(mean_squared_error(y_test,y_pred_test_ann))

print("Test : {:0.3f}".format(rmse))



model.save('Uber_ANN')
model_ann = load_model('Uber_ANN')
y_pred_test_ANN = model_ann.predict(X_test)

plt.plot(y_test, label='True')

plt.plot(y_pred_test_ANN, label='ANN')

plt.title("ANN's_Prediction")

plt.xlabel('Observation')

plt.ylabel('INR_Scaled')

plt.legend()

plt.show()
score_ann= model_ann.evaluate(X_test, y_test, batch_size=1)

print('ANN: %f'%score_ann)