import pandas as pd          
import numpy as np          # For mathematical calculations
import matplotlib.pyplot as plt  # For plotting graphs
from datetime import datetime    # To access datetime
from pandas import Series        # To work on series
%matplotlib inline
import warnings                   # To ignore the warnings
warnings.filterwarnings("ignore")
# Now let’s read the data
candies=pd.read_csv("../input/candy_production.csv")
candies_original=candies.copy()
candies.columns
candies.dtypes
candies.shape
candies.head()
candies.tail()
candies['observation_date'] = pd.to_datetime(candies.observation_date,format='%Y-%m-%d')  
candies_original['observation_date'] = pd.to_datetime(candies_original.observation_date,format='%Y-%m-%d')
#  let’s extract the year, month and day from the observation_date
for i in (candies,candies_original):
    i['year']=i.observation_date.dt.year 
    i['month']=i.observation_date.dt.month 
    i['day']=i.observation_date.dt.day
candies.head()
candies.index = candies['observation_date'] # indexing the Datetime to get the time period on the x-axis.
ts = candies['IPG3113N']
plt.figure(figsize=(16,8))
plt.plot(ts, label='% Candy Production')
plt.title('Candy Production')
plt.xlabel("Time(year)")
plt.ylabel("% Candy Production")
plt.legend(loc='best')
# let’s look at yearly production count.
plt.figure(figsize=(16,8))
candies.groupby('year')['IPG3113N'].mean().plot.bar()
# let’s look at monthly production count.
plt.figure(figsize=(16,8))
candies.groupby('month')['IPG3113N'].mean().plot.bar()
# Let’s look at the monthly mean of each year separately.

temp=candies.groupby(['year','month'])['IPG3113N'].mean()
temp.plot(figsize=(15,5), title= 'production Count(Monthwise)', fontsize=14)
train=candies.ix[:'2011-10-01']
valid=candies.ix['2011-11-01':]
train.head()
train.IPG3113N.plot(figsize=(15,8), title= 'Candy Production', fontsize=14, label='train')
valid.IPG3113N.plot(figsize=(15,8), title= 'Candy Production', fontsize=14, label='valid')
plt.xlabel("observation_date")
plt.ylabel("production count")
plt.legend(loc='best')
plt.show()
# predictions using naive approach for the validation set.
dd= np.asarray(train['IPG3113N'])
y_hat = valid.copy()
y_hat['naive'] = dd[len(dd)-1]
plt.figure(figsize=(12,8))
plt.plot(train.index, train['IPG3113N'], label='Train')
plt.plot(valid.index,valid['IPG3113N'], label='Valid')
plt.plot(y_hat.index,y_hat['naive'], label='Naive Forecast')
plt.legend(loc='best')
plt.title("Naive Forecast")
plt.show()
# RMSE(Root Mean Square Error) to check the accuracy of our model on validation data set.
from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(valid['IPG3113N'], y_hat.naive))
print(rms)
# last 5 observations.
y_hat_avg = valid.copy()
y_hat_avg['moving_avg_forecast'] = train['IPG3113N'].rolling(5).mean().iloc[-1] # average of last 5 observations.
plt.figure(figsize=(15,5)) 
plt.plot(train['IPG3113N'], label='Train')
plt.plot(valid['IPG3113N'], label='Valid')
plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast using 5 observations')
plt.legend(loc='best')
plt.show()
# last 7 observations.
y_hat_avg = valid.copy()
y_hat_avg['moving_avg_forecast'] = train['IPG3113N'].rolling(7).mean().iloc[-1] # average of last 7 observations.
plt.figure(figsize=(15,5)) 
plt.plot(train['IPG3113N'], label='Train')
plt.plot(valid['IPG3113N'], label='Valid')
plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast using 7 observations')
plt.legend(loc='best')
plt.show()
# last 50 observations.
y_hat_avg = valid.copy()
y_hat_avg['moving_avg_forecast'] = train['IPG3113N'].rolling(50).mean().iloc[-1] # average of last 50 observations.
plt.figure(figsize=(15,5)) 
plt.plot(train['IPG3113N'], label='Train')
plt.plot(valid['IPG3113N'], label='Valid')
plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast using 50 observations')
plt.legend(loc='best')
plt.show()
rms = sqrt(mean_squared_error(valid['IPG3113N'], y_hat_avg.moving_avg_forecast))
print(rms)
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
y_hat_ex = valid.copy()
fit2 = SimpleExpSmoothing(np.asarray(train['IPG3113N'])).fit(smoothing_level=0.6,optimized=False)
y_hat_ex['SES'] = fit2.forecast(len(valid))
plt.figure(figsize=(16,8))
plt.plot(train['IPG3113N'], label='Train')
plt.plot(valid['IPG3113N'], label='Valid')
plt.plot(y_hat_ex['SES'], label='SES')
plt.legend(loc='best')
plt.show()
rms = sqrt(mean_squared_error(valid['IPG3113N'], y_hat_ex['SES']))
print(rms)
import statsmodels.api as sm
sm.tsa.seasonal_decompose(train['IPG3113N']).plot()
result = sm.tsa.stattools.adfuller(train['IPG3113N'])
plt.show()
y_hat_ex = valid.copy()

fit1 = Holt(np.asarray(train['IPG3113N'])).fit(smoothing_level = 0.3,smoothing_slope = 0.1)
y_hat_ex['Holt_linear'] = fit1.forecast(len(valid))

plt.figure(figsize=(16,8))
plt.plot(train['IPG3113N'], label='Train')
plt.plot(valid['IPG3113N'], label='Valid')
plt.plot(y_hat_ex['Holt_linear'], label='Holt_linear')
plt.legend(loc='best')
plt.show()
rms = sqrt(mean_squared_error(valid['IPG3113N'], y_hat_ex.Holt_linear))
print(rms)
y_hat_win = valid.copy()
fit1 = ExponentialSmoothing(np.asarray(train['IPG3113N']) ,seasonal_periods=25 ,trend='add', seasonal='add',).fit()
y_hat_win['Holt_Winter'] = fit1.forecast(len(valid))
plt.figure(figsize=(16,8))
plt.plot( train['IPG3113N'], label='Train')
plt.plot(valid['IPG3113N'], label='Valid')
plt.plot(y_hat_win['Holt_Winter'], label='Holt_Winter')
plt.legend(loc='best')
plt.show()
rms = sqrt(mean_squared_error(valid['IPG3113N'], y_hat_win.Holt_Winter))
print(rms)
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(24).mean()
    rolstd = timeseries.rolling(24).std()
    
    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
plt.figure(figsize=(16,8))
test_stationarity(candies_original['IPG3113N'])
train_log = np.log(train['IPG3113N'])
valid_log = np.log(valid['IPG3113N'])

moving_avg = train_log.rolling(24).mean()
plt.figure(figsize=(16,8))
plt.plot(train_log)
plt.plot(moving_avg, color = 'red')
plt.show()
train_log_moving_avg_diff = train_log - moving_avg
train_log_moving_avg_diff.dropna(inplace = True)
plt.figure(figsize=(16,8))
test_stationarity(train_log_moving_avg_diff)
train_log_diff = train_log - train_log.shift(1)
plt.figure(figsize=(16,8))
test_stationarity(train_log_diff.dropna())
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(pd.DataFrame(train_log).IPG3113N.values, freq = 24)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.figure(figsize=(16,8))
plt.subplot(411)
plt.plot(train_log, label='Original')
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
# Let’s check stationarity of residuals.

train_log_decompose = pd.DataFrame(residual)
train_log_decompose['date'] = train_log.index
train_log_decompose.set_index('date', inplace = True)
train_log_decompose.dropna(inplace=True)
plt.figure(figsize=(16,8))
test_stationarity(train_log_decompose[0])
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(train_log_diff.dropna(), nlags=25)
lag_pacf = pacf(train_log_diff.dropna(), nlags=25, method='ols')
# ACF plot
plt.figure(figsize=(16,8))
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
plt.show()
# PACF plot
plt.figure(figsize=(16,8))
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.show()
from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(train_log, order=(2, 1, 0))  # here the q value is zero since it is just the AR model
results_AR = model.fit(disp=-1)  
plt.figure(figsize=(16,8))
plt.plot(train_log_diff.dropna(), label='original')
plt.plot(results_AR.fittedvalues, color='red', label='predictions')
plt.legend(loc='best')
plt.show()
# First step would be to store the predicted results as a separate series and observe it.
AR_predict=results_AR.predict(start="2011-11-01", end="2017-08-01")
AR_predict=AR_predict.cumsum().shift().fillna(0)
AR_predict1=pd.Series(np.ones(valid.shape[0]) * np.log(valid['IPG3113N'])[0], index = valid.index)
AR_predict1=AR_predict1.add(AR_predict,fill_value=0)
AR_predict = np.exp(AR_predict1)

plt.figure(figsize=(16,8))
plt.plot(valid['IPG3113N'], label = "Valid")
plt.plot(AR_predict, color = 'red', label = "Predict")
plt.legend(loc= 'best')
plt.title('RMSE: %.4f'% (np.sqrt(np.dot(AR_predict, valid['IPG3113N']))/valid.shape[0]))
plt.show()

model = ARIMA(train_log, order=(0, 1, 2))  # here the p value is zero since it is just the MA model
results_MA = model.fit(disp=-1)
plt.figure(figsize=(16,8))
plt.plot(train_log_diff.dropna(), label='original')
plt.plot(results_MA.fittedvalues, color='red', label='prediction')
plt.legend(loc='best')
plt.show()
MA_predict=results_MA.predict(start="2011-11-01", end="2017-08-01")
MA_predict=MA_predict.cumsum().shift().fillna(0)
MA_predict1=pd.Series(np.ones(valid.shape[0]) * np.log(valid['IPG3113N'])[0], index = valid.index)
MA_predict1=MA_predict1.add(MA_predict,fill_value=0)
MA_predict = np.exp(MA_predict1)

plt.figure(figsize=(16,8))
plt.plot(valid['IPG3113N'], label = "Valid")
plt.plot(MA_predict, color = 'red', label = "Predict")
plt.legend(loc= 'best')
plt.title('RMSE: %.4f'% (np.sqrt(np.dot(MA_predict, valid['IPG3113N']))/valid.shape[0]))
plt.show()
model = ARIMA(train_log, order=(2, 1, 2))  
results_ARIMA = model.fit(disp=-1)  

plt.figure(figsize=(16,8))
plt.plot(train_log_diff.dropna(),  label='original')
plt.plot(results_ARIMA.fittedvalues, color='red', label='predicted')
plt.legend(loc='best')
plt.show()
# Let’s define a function which can be used to change the scale of the model to the original scale.

def check_prediction_diff(predict_diff, given_set):
    predict_diff= predict_diff.cumsum().shift().fillna(0)
    predict_base = pd.Series(np.ones(given_set.shape[0]) * np.log(given_set['IPG3113N'])[0], index = given_set.index)
    predict_log = predict_base.add(predict_diff,fill_value=0)
    predict = np.exp(predict_log)
    
    plt.figure(figsize=(16,8))
    plt.plot(given_set['IPG3113N'], label = "Given set")
    plt.plot(predict, color = 'red', label = "Predict")
    plt.legend(loc= 'best')
    plt.title('RMSE: %.4f'% (np.sqrt(np.dot(predict, given_set['IPG3113N']))/given_set.shape[0]))
    plt.show()
def check_prediction_log(predict_log, given_set):
    predict = np.exp(predict_log)
    
    plt.figure(figsize=(16,8))
    plt.plot(given_set['IPG3113N'], label = "Given set")
    plt.plot(predict, color = 'red', label = "Predict")
    plt.legend(loc= 'best')
    plt.title('RMSE: %.4f'% (np.sqrt(np.dot(predict, given_set['IPG3113N']))/given_set.shape[0]))
    plt.show()
ARIMA_predict_diff=results_ARIMA.predict(start="2011-11-01", end="2017-08-01")

check_prediction_diff(ARIMA_predict_diff, valid)
from statsmodels.tsa.statespace.sarimax import SARIMAX

y_hat_avg = valid.copy()
fit1 = SARIMAX(train['IPG3113N'], order=(2, 1, 4),seasonal_order=(0,1,1,7),enforce_stationarity=False,enforce_invertibility=False).fit()
y_hat_ex['SARIMA'] = fit1.predict(start="2011-11-01", end="2017-08-01", dynamic=True)
plt.figure(figsize=(16,8))
plt.plot( train['IPG3113N'], label='Train')
plt.plot(valid['IPG3113N'], label='Valid')
plt.plot(y_hat_ex['SARIMA'], label='SARIMA')
plt.legend(loc='best')
plt.show()
# Let’s check the rmse value for the validation part.

rms = sqrt(mean_squared_error(valid['IPG3113N'], y_hat_ex.SARIMA))
print(rms)