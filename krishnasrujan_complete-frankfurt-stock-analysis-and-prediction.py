import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import time
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.arima_model import ARIMA
# Simple Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn import metrics
#pd.set_option('display.max_columns',None)
#pd.set_option('display.max_rows',None)
#Frankfurt stock exchange
df=pd.read_csv('/kaggle/input/frankfurt-stock-exchange/frankfurt_stock_exchange.csv',parse_dates=['Date'])#,index_col='Date')
print(df.info())
df.head()
df.drop(columns=['Change','Last Price of the Day','Daily Traded Units','Daily Turnover','Turnover','Traded Volume'],inplace=True)
df.head()
data = df
for i in ['Open','Close','High','Low']:
    data[i] = data[i].ffill()
data
data = data[169:]
data['year'] = pd.DatetimeIndex(data['Date']).year
data['month'] = [d.strftime('%b') for d in data.Date]
data['day'] = pd.DatetimeIndex(data['Date']).dayofweek
data.set_index('Date',inplace=True)
data
# Draw Plot
fig, axes = plt.subplots(3, 1, figsize=(20,15), dpi= 80)
sns.boxplot(x='year', y='Close', data=data, ax=axes[0])
sns.boxplot(x='month', y='Close', data=data,ax = axes[1])
sns.boxplot(x='day', y='Close', data=data,ax = axes[2])

# Set Title
axes[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize=18); 

axes[1].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize=18)

axes[2].set_title('Day-wise Box Plot\n(The Seasonality)', fontsize=18)
plt.show()
plt.plot(data['Close'])
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse

# Import Data
# Multiplicative Decomposition 
#result_mul = seasonal_decompose(data['Total_Sunspots'], model='multiplicative', extrapolate_trend='freq')

# Additive Decomposition
result_add = seasonal_decompose(data['Close'], model='add', extrapolate_trend='freq',freq=5)

# Plot
plt.rcParams.update({'figure.figsize': (5,5)})
#result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)
result_add.plot().suptitle('Additive Decompose', fontsize=22)
plt.show()
data['rolling_mean'] = data['Close'].rolling(12).mean()
data['Detrend'] = data['Close'] - data['rolling_mean']
plt.plot(data['rolling_mean'])
plt.title('Trend',fontsize=16)
from statsmodels.tsa.seasonal import seasonal_decompose
result_add = seasonal_decompose(data['Close'], model='mul', extrapolate_trend='freq',freq=96)
deseasonalized = data.Close.values -result_add.seasonal
plt.plot(deseasonalized)
plt.title('Drug Sales deseasonalized by subtracting the seasonal component', fontsize=16)
from pandas.plotting import autocorrelation_plot


# Draw Plot
plt.rcParams.update({'figure.figsize':(9,5), 'figure.dpi':120})
autocorrelation_plot(data['Close'].tolist())
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Calculate ACF and PACF upto 50 lags
# acf_50 = acf(df.value, nlags=50)
# pacf_50 = pacf(df.value, nlags=50)

# Draw Plot
fig, axes = plt.subplots(1,2,figsize=(16,3), dpi= 100)
plot_acf(data['Close'].tolist(), lags=200, ax=axes[0])
plot_pacf(data['Close'].tolist(), lags=50, ax=axes[1])
count = int(data.shape[0]*0.8)
Train = data[:count]
Test = data[count:]

y_hat_avg = Test.copy()
fit1 = Holt(np.asarray(Train['Close'])).fit()
y_hat_avg['Holt_Winter'] = fit1.predict(start=count+1,end=data.shape[0])
plt.figure(figsize=(16,8))
plt.plot(Train.index, Train['Close'], label='Train')
plt.plot(Test.index,Test['Close'], label='Test')
plt.plot(y_hat_avg.index,y_hat_avg['Holt_Winter'], label='Holt_Winter')
plt.legend(loc='best')
new = data[['Close']]
Train, Test = new.iloc[:count,0], new.iloc[count:,0]
history = [x for x in Train]
predictions = []
lower_list = []
upper_list = []
for t in range(len(Test)):
    model = ARIMA(history, order=(2,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    lower = output[2][0][0]
    upper = output[2][0][1]
    predictions.append(yhat)
    lower_list.append(lower)
    upper_list.append(upper)
    obs = Test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
error = metrics.mean_squared_error(Test, predictions)
print('Test MSE: %.3f' % error)
# plot
plt.plot(Test.values,color='black')
plt.plot(lower_list,color='red')
plt.plot(upper_list,color='green')
plt.plot(predictions)
plt.show()
new = data[['Close']]
new.reset_index(inplace=True)
new.columns = ['ds','y']
new
new.shape
from fbprophet import Prophet
m = Prophet()
model = m.fit(new)
future = m.make_future_dataframe(periods=30)
future.tail()
future.shape, new.shape
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
forecast['yhat'][:2800].shape
new['y'].shape
fig1 = m.plot(forecast)
data['HL_PCT']=(data['High']-data['Close'])/data['Close']*100
data['PCT_change']=(data['Close']-data['Open'])/data['Open']*100
data.drop(['year','day','rolling_mean','Detrend','Open','High','Low','month'],axis=1,inplace=True)
data
forecast_out=4
data['forecast_col'] = data['Close'].shift(-forecast_out)
data.dropna(inplace=True)
data
train_count =int(data.shape[0]*0.85)
test_count = data.shape[0]-train_count
y_train = data.iloc[:train_count,3]
y_test  = data.iloc[train_count:,3]
print(y_train.shape,y_test.shape)
x_train = data.iloc[:train_count,0:-1]
x_test = data.iloc[train_count:,0:-1]
x_lately=x_train[-forecast_out:]
x_train=x_train[:-forecast_out]
df.dropna(inplace=True)
y_train=y_train[:-forecast_out]
print(y_train.shape)
x_train
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor()
clf.fit(x_train,y_train)
y_preds = clf.predict(x_test)
y_lately = clf.predict(x_lately)
plt.plot(y_preds,color='red')
plt.plot(y_test)
plt.show()
from sklearn import metrics
print(metrics.mean_squared_error(y_preds,y_test))
clf = RandomForestRegressor(n_estimators = 100,min_samples_leaf = 3,min_samples_split = 8)
clf.fit(x_train,y_train)
y_preds = clf.predict(x_test)
y_lately = clf.predict(x_lately)
plt.plot(y_preds,color='red')
plt.plot(y_test)
plt.show()
from sklearn import metrics
print(metrics.mean_squared_error(y_preds,y_test))
