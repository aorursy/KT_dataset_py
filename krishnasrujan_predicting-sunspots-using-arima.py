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
data = pd.read_csv('/kaggle/input/sunspots/Sunspots.csv',index_col=0,#parse_dates=['Date'],
                   names=['Date','Total_Sunspots'],header=0)
data.tail(100)
data = data.loc[(data['Date']>='1751') & (data['Date']<='2001'),['Date','Total_Sunspots']]
data.reset_index(inplace=True)
data.drop('index',axis=1,inplace=True)
data
data['nth_year'] = [str(d)[3] for d in data.Date]
data['nth_year']=data['nth_year'].replace('0','10')
data
data['Date'] = pd.to_datetime(data['Date'])
data['nth_year'] = data['nth_year'].astype(float)
plt.plot(data['Total_Sunspots'])
data['year'] = pd.DatetimeIndex(data['Date']).year
data['month'] = [d.strftime('%b') for d in data.Date]
data
fig, axes = plt.subplots(3, 1, figsize=(20,15), dpi= 80)
sns.boxplot(x='year', y='Total_Sunspots', data=data, ax=axes[0])
sns.boxplot(x='nth_year', y='Total_Sunspots', data=data,ax = axes[1])
sns.boxplot(x='month', y='Total_Sunspots', data=data,ax = axes[2])

# Set Title
axes[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize=18); 

axes[1].set_title('Nth_year-wise Box Plot\n(The Seasonality)', fontsize=18)

axes[2].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize=18)
plt.show()
data.set_index('Date',inplace=True)
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse

# Import Data
# Multiplicative Decomposition 
#result_mul = seasonal_decompose(data['Total_Sunspots'], model='multiplicative', extrapolate_trend='freq')

# Additive Decomposition
result_add = seasonal_decompose(data['Total_Sunspots'], model='additive', extrapolate_trend='freq',freq=96)

# Plot
plt.rcParams.update({'figure.figsize': (5,5)})
#result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)
result_add.plot().suptitle('Additive Decompose', fontsize=22)
plt.show()
data['rolling_mean'] = data['Total_Sunspots'].rolling(12).mean()
data['Detrend'] = data['Total_Sunspots'] - data['rolling_mean']
data
plt.plot(data['Detrend'])
plt.title('Detrended',fontsize=16)
# Using statmodels: Subtracting the Trend Component.
from statsmodels.tsa.seasonal import seasonal_decompose
result_add = seasonal_decompose(data['Total_Sunspots'], model='add', extrapolate_trend='freq',freq=96)
detrended = data.Total_Sunspots.values -result_add.trend
plt.plot(detrended)
plt.title('Drug Sales detrended by subtracting the trend component', fontsize=16)
from statsmodels.tsa.seasonal import seasonal_decompose
result_add = seasonal_decompose(data['Total_Sunspots'], model='add', extrapolate_trend='freq',freq=96)
deseasonalized = data.Total_Sunspots.values -result_add.seasonal
plt.plot(deseasonalized)
plt.title('Drug Sales deseasonalized by subtracting the seasonal component', fontsize=16)
from pandas.plotting import autocorrelation_plot


# Draw Plot
plt.rcParams.update({'figure.figsize':(9,5), 'figure.dpi':120})
autocorrelation_plot(data['Total_Sunspots'].tolist())
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Draw Plot
fig, axes = plt.subplots(1,2,figsize=(16,3), dpi= 100)
plot_acf(data['Total_Sunspots'].tolist(), lags=100, ax=axes[0])
plot_pacf(data['Total_Sunspots'].tolist(), lags=50, ax=axes[1])
count = int(data.shape[0]*0.8)
Train = data[:count]
Test = data[count:]

y_hat_avg = Test.copy()
fit1 = Holt(np.asarray(Train['Total_Sunspots'])).fit()
y_hat_avg['Holt_Winter'] = fit1.predict(start=count+1,end=data.shape[0])
plt.figure(figsize=(16,8))
plt.plot(Train.index, Train['Total_Sunspots'], label='Train')
plt.plot(Test.index,Test['Total_Sunspots'], label='Test')
plt.plot(y_hat_avg.index,y_hat_avg['Holt_Winter'], label='Holt_Winter')
plt.legend(loc='best')
# plt.savefig('Holt_Winters.jpg')
new = data[['Total_Sunspots']]
new
new.index.freq = 'M' # Start of the month
Train, Test = new.iloc[:count, 0], data.iloc[count:, 0]

model = ExponentialSmoothing(Train, trend='add', seasonal='add', seasonal_periods=12, damped=True)
hw_model = model.fit(optimized=True, use_boxcox=False, remove_bias=False)
pred = hw_model.predict(start=Test.index[0], end=Test.index[-1])

plt.plot(Train.index, Train, label='Train')
plt.plot(Test.index, Test, label='Test')
plt.plot(pred.index, pred, label='Holt-Winters')
plt.legend(loc='best');
history = [x for x in Train]
predictions = []
lower_list = []
upper_list = []
for t in range(len(Test)):
    model = ARIMA(history, order=(5,0,1))
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
