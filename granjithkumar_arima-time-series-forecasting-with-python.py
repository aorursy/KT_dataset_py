import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('../input/new2021/new (2).csv')
data.head(2)
data.columns
data.drop(labels = ['Unnamed: 0'],axis = 1,inplace =True)
data.describe()
data['Date']=pd.to_datetime(data['Date']) #Converting date into datetime object
data_new = data.set_index(data['Date']) #Setting the date column as index
data_new1 = data_new.drop(labels =['Date'],axis = 1) #Deleting the data column
fig = plt.figure(figsize = (10,5))
data_new1['Monthly Mean Total Sunspot Number'].plot(style = 'k.')
data_new1['2019'].resample('M').mean().plot(kind='bar')
data_q = data_new1.resample('q').mean()
data_q.head()
#Ho: It is non stationary
#H1: It is stationary
def adfuller_test(data):
    result = adfuller(data)
    labels =['ADF Tesr Statistic','p-value','Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+":"+str(value))
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
adfuller_test(data_q)
data_q.plot()
base_data = data_q.copy()
base_data['Shifted_Monthly Mean Total Sunspot Number'] = base_data['Monthly Mean Total Sunspot Number'].shift(1)
base_data[['Monthly Mean Total Sunspot Number','Shifted_Monthly Mean Total Sunspot Number']].plot(figsize=(12,8))
base_data[['Monthly Mean Total Sunspot Number','Shifted_Monthly Mean Total Sunspot Number']]['2018':].plot(figsize=(12,8))
base_data = base_data.dropna()
from sklearn.metrics import mean_squared_error
print('Mean Squared Error: '+str(mean_squared_error(base_data['Monthly Mean Total Sunspot Number'],base_data['Shifted_Monthly Mean Total Sunspot Number'])))
fig = plt.figure(figsize = (10,10))
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(data_q)
plt.show()
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(data_q,lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(data_q,lags=40,ax=ax2)

from statsmodels.tsa.arima_model import ARIMA
model=ARIMA(data_q,order=(2,0,2))
model_fit=model.fit()
model_fit.summary()
data_q['forecast']=model_fit.predict(start=1000,end=1500,dynamic=True)
data_q[['Monthly Mean Total Sunspot Number','forecast']].plot(figsize=(12,8))
pred = data_q[data_q.forecast.notna()]
pred[['Monthly Mean Total Sunspot Number','forecast']].plot(figsize=(12,8))
model=sm.tsa.statespace.SARIMAX(data_q['Monthly Mean Total Sunspot Number'],order=(2, 0, 2),seasonal_order=(2,0,2,6)) #seasonal_order is (p,d,q,seasonal_value) In this case I'm considering it as 6
results=model.fit()
results.summary()
data_q['forecast']=results.predict(start=1000,end=1084,dynamic=True)
data_q[['Monthly Mean Total Sunspot Number','forecast']].plot(figsize=(12,8))
pred = data_q[data_q.forecast.notna()]
pred[['Monthly Mean Total Sunspot Number','forecast']].plot(figsize=(12,8))