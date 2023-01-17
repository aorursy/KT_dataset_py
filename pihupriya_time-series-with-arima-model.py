# importing all libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import datetime
train =  pd.read_csv("C:/Users/Acer/Desktop/Time Series/predice-el-futuro/train_csv.csv")
train.head()    #printing first five values
test = pd.read_csv("C:/Users/Acer/Desktop/Time Series/predice-el-futuro/test_csv.csv")
test.head()
train.dtypes
train.info()
train['datetime'] = pd.to_datetime(train['time'],format='%Y-%m-%d %H:%M:%S')
train.head()
#Date Related Features
train['year']=train['datetime'].dt.year 
train['month']=train['datetime'].dt.month 
train['day']=train['datetime'].dt.day
train['dayofweek_num']=train['datetime'].dt.dayofweek  
train['dayofweek_name']=train['datetime'].dt.weekday_name
train.head()
#Time based Features
train['hour'] = train['datetime'].dt.hour
train['minute'] = train['datetime'].dt.minute
train['second'] = train['datetime'].dt.second
train.head()
#Lag Features
train['lag_1'] = train['feature'].shift(1)        #Previous feature is important to make prediction.The value at time t is greatly affected by the value of t-1     
train['lag_2'] = train['feature'].shift(2)        # The past values are known by Lags  
train['lag_3'] = train['feature'].shift(3)

train.head(10)
#Rolling window feature
train['rolling_mean_3'] = train['feature'].rolling(window=3).mean()

train.head(10)
train['rolling_mean_6'] = train['feature'].rolling(window=6).mean()

train.head(10)
train['rolling_mean_9'] = train['feature'].rolling(window=9).mean()
#train_1 = train[['datetime', 'rolling_mean_3', 'feature']]
train.head(10)
train['rolling_mean_12'] = train['feature'].rolling(window=12).mean()

train.head(10)
train['expanding_mean'] = train['feature'].expanding(2).mean()

train.head(10)
import datetime
#Statistical Test
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()
    
    dftest = adfuller(train['feature'])

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    print('Results of Dickey-Fuller Test ')
    
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistics','p-value','#lags used','Number of observations used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value(%s)'%key] = value
    print(dfoutput)
test_stationarity(train['feature'])
#from datetime import datetime

#from statsmodels.tsa.stattools import adfuller
#from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

# Original Series
fig, axes = plt.subplots(3, 2, sharex=True)                     #With 3 rows and 2 columns(3,2)         
axes[0, 0].plot(train.feature); axes[0, 0].set_title('Original Series')       #with axes 0(0,0)
plot_acf(train.feature, ax=axes[0, 1])                                        #1st row 1st column(0,1)


# 1st Differencing          The purpose of differencing is to make the time series Stationary
axes[1, 0].plot(train.feature.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(train.feature.diff().dropna(), ax=axes[1, 1])


# 2nd Differencing
axes[2, 0].plot(train.feature.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(train.feature.diff().diff().dropna(), ax=axes[2, 1])

# 3rd Differencing
#axes[3, 0].plot(train.feature.diff().diff()); axes[3, 0].set_title('3rd Order Differencing')
#plot_acf(train.feature.diff().diff().dropna(), ax=axes[3, 1])


plt.show()


# PACF plot of 1st differenced series
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})
fig, axes = plt.subplots(1, 2, sharex=True)  #1 row 2 columns
axes[0].plot(train.feature.diff()); axes[0].set_title('1st Differencing')  #1st box
axes[1].set(ylim=(0,4))   #0 to 5
plot_pacf(train.feature.diff().dropna(), ax=axes[1])


plt.show()

#The PACF lag 1 is above the signifucance line. Fix p as 1
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})
fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(train.feature.diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,1.2))
plot_acf(train.feature.diff().dropna(), ax=axes[1])

#The ACF tells how many MA terms
#You can observe the ACF only lag 1 is quite above the significance line(blue region)  
#Fix q as 1
# 1,1,2 ARIMA Model
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(train.feature, order=(1,1,2))
model_fit = model.fit()  #disp=0
print(model_fit.summary())
# Actual vs Fitted
plt.rcParams.update({'figure.figsize':(9,2), 'figure.dpi':120})
model_fit.plot_predict(dynamic=False)
plt.show()
# Out-of-Time cross-validation,need to create the training and testing dataset by splitting the time series into 2 contiguous parts in approximately 75:25 ratio or a reasonable proportion based on time frequency of series.
training = train.feature[0:40]
testing = train.feature[40:]
training,testing
#training.plot( title= 'All day features', fontsize=7) 
#testing.plot( title= 'All day features', fontsize=7) 
#plt.show()
# Build Model 
model = ARIMA(training, order=(1, 1, 1))  
fitted = model.fit(disp=0)  

fc, se, conf = fitted.forecast(40, alpha=0.05)  # 80% conf

# Make as pandas series
fc_series = pd.Series(fc, index=testing.index)
lower_series = pd.Series(conf[:, 0], index=testing.index)
upper_series = pd.Series(conf[:, 1], index=testing.index)
plt.figure(figsize=(12,5), dpi=100)
plt.plot(training, label='training')
plt.plot(testing, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)

plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


model = ARIMA(training, order=(2, 1,1))  
fitted = model.fit(disp=-1)  
print(fitted.summary())

# Forecast
fc, se, conf = fitted.forecast(40, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=testing.index)
lower_series = pd.Series(conf[:, 0], index=testing.index)
upper_series = pd.Series(conf[:, 1], index=testing.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(training, label='training')
plt.plot(testing, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()
# Accuracy metrics
def forecast_accuracy(forecast, actual):
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    return({ 'rmse':rmse})

forecast_accuracy(fc, testing)
# Accuracy metrics
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    acf1 = acf(fc-testing)[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'acf1':acf1, 
            'corr':corr, 'minmax':minmax})

forecast_accuracy(fc, testing)

train.to_csv("Final_Set.csv", index = False)