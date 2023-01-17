# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('/kaggle/input/jhts-forecasting/train_6BJx641.csv',index_col=False)
train_df.name = 'train_df'
train_df.drop(columns=['var1','var2'],inplace=True)
train_df.head()
#NO Null data
train_df.isnull().sum()
print(train_df.columns)
print(train_df.dtypes)
from datetime import datetime
def df_preprocess(df):
    print(df.name)
#1
    df['datetime'] =  pd.to_datetime(df['datetime'], infer_datetime_format=True)
#2
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['date'] = df['datetime'].dt.day
    df['time'] = df['datetime'].dt.time
#3
    #df.drop(['datetime'],axis=1,inplace=True)
#4
    if (df.name == 'test_df'):
        df = df[['ID','datetime','year','month','date','time','temperature','pressure','windspeed']]
    else:
        df = df[['ID','datetime','year','month','date','time','temperature','pressure','windspeed','electricity_consumption']]
 #5
    df.head()
    return df
train_df = df_preprocess(train_df)
train_df.head()
import matplotlib.pyplot as plt
pd.plotting.register_matplotlib_converters()
%matplotlib inline
import seaborn as sns
# Set the width and height of the figure
plt.figure(figsize=(16,8))
# Add title
plt.title("Electricity Consumption Rate on yearly basis")
# Line chart showing yearly trend
sns.lineplot(y=train_df['electricity_consumption'], x=train_df['datetime'], label="Timeline")
# Set the width and height of the figure
plt.figure(figsize=(16,8))
# Add title
plt.title("Electricity Consumption Rate on yearly basis")
# Line chart showing yearly trend
sns.lineplot(y=train_df['electricity_consumption'], x=train_df['year'], label="Yearly")
# Set the width and height of the figure
plt.figure(figsize=(16,8))
# Add title
plt.title("Electricity Consumption Rate on monthly basis")
# Line chart showing monthly trend
sns.lineplot(y=train_df['electricity_consumption'], x=train_df['month'], label="Monthly")
# Set the width and height of the figure
plt.figure(figsize=(16,8))
# Add title
plt.title("Electricity Consumption Rate on daily basis")
# Line chart showing daily trend
sns.lineplot(y=train_df['electricity_consumption'], x=train_df['date'], label="Daily")
# Set the width and height of the figure
plt.figure(figsize=(16,8))
# Add title
plt.title("Electricity Consumption Rate variation on Temperature")
# Line chart showing temperature trend
sns.lineplot(y=train_df['electricity_consumption'], x=train_df['temperature'], label="Temperature")
# Set the width and height of the figure
plt.figure(figsize=(16,8))
# Add title
plt.title("Electricity Consumption Rate variation on Pressure")
# Line chart showing pressure trend
sns.lineplot(y=train_df['electricity_consumption'], x=train_df['pressure'], label="Pressure")
# Set the width and height of the figure
plt.figure(figsize=(16,8))
# Add title
plt.title("Electricity Consumption Rate variation on Windspeed")
# Line chart showing windspeed trend
sns.lineplot(y=train_df['electricity_consumption'], x=train_df['windspeed'], label="Windspeed")
train_df_lst = []

for i in range(1,13):
    train_df_lst.append('train_df_'+str(i))

print(train_df_lst)
    
for i in range(0,12):
    train_df_lst[i] = train_df[train_df["month"] == (i+1)]    
# Set the width and height of the figure
plt.figure(figsize=(16,8))
# Add title
plt.title("Electricity Consumption Rate on yearly basis")
# Line chart showing yearly trend
for i in range(len(train_df_lst)):
    sns.lineplot(y=train_df_lst[i]['electricity_consumption'], x=train_df_lst[i]['year'], label=i)
# Set the width and height of the figure
plt.figure(figsize=(16,8))
# Add title
plt.title("Electricity Consumption Rate on daily basis")
# Line chart showing yearly trend
for i in range(len(train_df_lst)):
    sns.lineplot(y=train_df_lst[i]['electricity_consumption'], x=train_df_lst[i]['date'], label=i)
train_df_lst[0]['year'].value_counts()
ts_data_1_lst = []
for i in range(train_df_lst[0]['year'].nunique()):
    ts_data_1_lst.append('ts_data_1_lst_'+str(i+1))

print(ts_data_1_lst)
    
for i in range(train_df_lst[0]['year'].nunique()):
    ts_data_1_lst[i] = train_df_lst[0][train_df_lst[0]["year"] == (2013+1)]    
    
ts_data_1_lst[0].head()
ts_data_1_lst[0].shape
ts_data_1_lst_1 = ts_data_1_lst[0].drop(['ID','year','month','date','time','temperature','pressure','windspeed'],axis=1)
ts_data_1_lst_1 = ts_data_1_lst_1.set_index(['datetime'])
ts_data_1_lst_1.head()
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):
    plt.figure(figsize=(16,8))
    
    ##Determining Rolling Statistics
    rollingmean = timeseries.rolling(window=24).mean()
    rollingstd = timeseries.rolling(window=24).std()
    
    ##Plot Rolling Statistics
    orig = plt.plot(timeseries, color = 'blue',label = 'Original')
    mean = plt.plot(rollingmean, color = 'red',label = 'Rolling Mean')
    std = plt.plot(rollingstd, color = 'black',label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show(block=False)

    ##Perform Dickey Fuller Test
    print('Results of Dickey Fuller Test:')
    dftest = adfuller(timeseries['electricity_consumption'], autolag = 'AIC')
    dfoutput = pd.Series(dftest[0:4], index = ['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
test_stationarity(ts_data_1_lst_1)
#Applying log on the dataset - to reduce the variability in the data
ts_data_1_lst_1_logscale = np.log(ts_data_1_lst_1)
ts_data_1_lst_1_logscale.head()
test_stationarity(ts_data_1_lst_1_logscale)
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_data_1_lst_1_logscale)

trend=decomposition.trend
seasonal=decomposition.seasonal
residual=decomposition.resid
plt.figure(figsize=(16,8))
plt.subplot(411)
plt.plot(ts_data_1_lst_1_logscale, label='Original')
#plt.plot(ts_data_1_lst_1, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='Seasonal')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residual')
plt.legend(loc='best')
plt.tight_layout()
#Residual
decomposed_ts_data_1_lst_1_logscale = pd.DataFrame(residual)
decomposed_ts_data_1_lst_1_logscale.columns = ts_data_1_lst_1_logscale.columns
decomposed_ts_data_1_lst_1_logscale.dropna(inplace=True)
test_stationarity(decomposed_ts_data_1_lst_1_logscale)
#Seasonality
seasonal_ts_data_1_lst_1_logscale = pd.DataFrame(seasonal)
seasonal_ts_data_1_lst_1_logscale.columns = ts_data_1_lst_1_logscale.columns
seasonal_ts_data_1_lst_1_logscale.dropna(inplace=True)
test_stationarity(seasonal_ts_data_1_lst_1_logscale)
#Trend
trend_ts_data_1_lst_1_logscale = pd.DataFrame(trend)
trend_ts_data_1_lst_1_logscale.columns = ts_data_1_lst_1_logscale.columns
trend_ts_data_1_lst_1_logscale.dropna(inplace=True)
test_stationarity(trend_ts_data_1_lst_1_logscale)
ts_data_1_lst_1_logscale
ts_data_1_lst_1_logscale.shift()
ts_data_1_lst_1_shift = ts_data_1_lst_1_logscale - ts_data_1_lst_1_logscale.shift()
#ts_data_1_lst_1_shift = ts_data_1_lst_1 - ts_data_1_lst_1.shift()
ts_data_1_lst_1_shift.dropna(inplace=True)
ts_data_1_lst_1_shift
test_stationarity(ts_data_1_lst_1_shift)
from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(ts_data_1_lst_1_shift, nlags=20)
lag_pacf = pacf(ts_data_1_lst_1_shift, nlags=20, method = 'ols')

plt.figure(figsize=(16,8))

#Plot ACF
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='grey')
plt.axhline(y=-1.96/np.sqrt(len(ts_data_1_lst_1_shift)),linestyle='--',color='grey')
plt.axhline(y=1.96/np.sqrt(len(ts_data_1_lst_1_shift)),linestyle='--',color='grey')
plt.title('Auto Correlation Function')
plt.tight_layout()

#Plot PACF
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='grey')
plt.axhline(y=-1.96/np.sqrt(len(ts_data_1_lst_1_shift)),linestyle='--',color='grey')
plt.axhline(y=1.96/np.sqrt(len(ts_data_1_lst_1_shift)),linestyle='--',color='grey')
plt.title('Partial Auto Correlation Function')
plt.tight_layout()
ts_data_1_lst_1_logscale
from statsmodels.tsa.arima_model import ARIMA

#ts_data_1_lst_1_shift

#ARIMA Model
model = ARIMA(ts_data_1_lst_1_logscale, order=(1,1,1))
# order=(1,1,1) - order=(P,d,Q)
results_ARIMA = model.fit(disp=-1)
plt.plot(ts_data_1_lst_1_shift)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f' %sum((results_ARIMA.fittedvalues - ts_data_1_lst_1_shift["electricity_consumption"])**2))
## RSS - Residual Sum of Squares
print('Plotting ARIMA Model')
prediction_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues,copy=True)
print(prediction_ARIMA_diff.head())
#Calculating cumulated sum ( x(t) = x(t) + x(t-1) )
prediction_ARIMA_diff_cumsum = prediction_ARIMA_diff.cumsum()
print(prediction_ARIMA_diff_cumsum.head())
ts_data_1_lst_1_logscale['electricity_consumption'].iloc[0:]
prediction_ARIMA_log = pd.Series(ts_data_1_lst_1_logscale['electricity_consumption'].iloc[0:], index=ts_data_1_lst_1_logscale.index)
prediction_ARIMA_log = prediction_ARIMA_log.add(prediction_ARIMA_diff_cumsum,fill_value=0)
prediction_ARIMA_log.head()
plt.figure(figsize=(16,8))
prediction_ARIMA = np.exp(prediction_ARIMA_log)
plt.plot(ts_data_1_lst_1)
plt.plot(prediction_ARIMA)
prediction_ARIMA = prediction_ARIMA.round(1)
plt.figure(figsize=(16,8))
#Prediction from 24th jan till 31st jan - 8 days* 24 hrs = 192 hrs
results_ARIMA.plot_predict(1,(552+192))
from sklearn.metrics import mean_absolute_error,mean_squared_error
from math import sqrt
mae = mean_absolute_error(ts_data_1_lst_1['electricity_consumption'], prediction_ARIMA).round(1)
mse = mean_squared_error(ts_data_1_lst_1['electricity_consumption'], prediction_ARIMA).round(1)
print('MAE: %f' % mae)
print('MSE: %f' % mse)
print('RMSE: %f' % sqrt(mse))
#Print Error list:
error = []
for i in range(len(ts_data_1_lst_1)):
    error.append(abs(ts_data_1_lst_1['electricity_consumption'][i] - prediction_ARIMA[i]).round(1))
print(error)    
#Forecast for future 8 days - converting into df for better display
pred_janEOM = pd.DataFrame(results_ARIMA.forecast(192))

#The forecast function returns 3 ARRAYS
#1 forecastndarray - Array of out of sample forecasts
#2 stderrndarray - Array of the standard error of the forecasts.
#3 conf_intndarray - 2d array of the confidence interval for the forecast
#
pred_janEOM
#Transpose the df
pred_janEOM_t = pred_janEOM.T
print(pred_janEOM_t)
#Convert to forecasted value to exponent, as the fittng was done in log scale
pred_janEOM_fcv = pred_janEOM_t[0].astype('float64')
pred_janEOM_fcv_exp = np.exp(pred_janEOM_fcv).round(1)
pred_janEOM_fcv_exp
test_df = pd.read_csv('/kaggle/input/jhts-forecasting/test_pavJagI.csv',index_col=False)
test_df.name = 'test_df'
test_df.drop(columns=['var1','var2'],inplace=True)
test_df.head()
print(test_df.isnull().sum())
print(test_df.columns)
print(test_df.dtypes)
test_df = df_preprocess(test_df)

test_df_lst = []

for i in range(1,13):
    test_df_lst.append('test_df_'+str(i))

print(test_df_lst)
    
for i in range(0,12):
    test_df_lst[i] = test_df[test_df["month"] == (i+1)]  

tst_data_1_lst = []
for i in range(test_df_lst[0]['year'].nunique()):
    tst_data_1_lst.append('tst_data_1_lst_'+str(i+1))

print(tst_data_1_lst)
    
for i in range(test_df_lst[0]['year'].nunique()):
    tst_data_1_lst[i] = test_df_lst[0][test_df_lst[0]["year"] == (2013+1)]    
    
tst_data_1_lst[0].head()
tst_data_1_lst_1 = tst_data_1_lst[0].drop(['ID'],axis=1)
tst_data_1_lst_1 = tst_data_1_lst_1.set_index(['datetime'])
tst_data_1_lst_1.head()
tst_data_1_lst_1['electricity_consumption'] = pred_janEOM_fcv_exp.values
tst_data_1_lst_1
