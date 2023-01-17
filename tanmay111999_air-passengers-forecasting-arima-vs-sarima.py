import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from itertools import combinations

from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA as ARIMA
import statsmodels.api as sm
import statsmodels.tsa.api as smt
pd.options.display.float_format = '{:.2f}'.format
data = pd.read_csv('../input/air-passengers/AirPassengers.csv')
data.head()
data.isnull().sum()
data.info()
data['Date'] = pd.to_datetime(data['Month'])
data = data.drop(columns = 'Month')
data = data.set_index('Date')
data = data.rename(columns = {'#Passengers':'Passengers'})
data.head()
def test_stationarity(timeseries):
    #Determing rolling statistics
    MA = timeseries.rolling(window=12).mean()
    MSTD = timeseries.rolling(window=12).std()

    #Plot rolling statistics:
    plt.figure(figsize=(15,5))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(MA, color='red', label='Rolling Mean')
    std = plt.plot(MSTD, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        
        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()
test_stationarity(data['Passengers'])
dec = sm.tsa.seasonal_decompose(data['Passengers'],period = 12).plot()
plt.show()
sns.distplot(data['Passengers'])
log_data = np.log(data)
log_data.head()
test_stationarity(log_data['Passengers'])
sns.distplot(log_data['Passengers'])
data_diff = data['Passengers'].diff()
data_diff = data_diff.dropna()
dec = sm.tsa.seasonal_decompose(data_diff,period = 12).plot()
plt.show()
test_stationarity(data_diff)
log_data_diff = log_data['Passengers'].diff()
log_data_diff = log_data_diff.dropna()
dec = sm.tsa.seasonal_decompose(log_data_diff,period = 12)
dec.plot()
plt.show()
test_stationarity(log_data_diff)
tsplot(data_diff)
model = ARIMA(data['Passengers'],order = (2,1,2))
model_fit = model.fit()
print(model_fit.summary())
data['FORECAST'] = model_fit.predict(start = 120,end = 144,dynamic = True)
data[['Passengers','FORECAST']].plot(figsize = (10,6))
exp = [data.iloc[i,0] for i in range(120,len(data))]
pred = [data.iloc[i,1] for i in range(120,len(data))]
data = data.drop(columns = 'FORECAST')
print(mean_absolute_error(exp,pred))
data_diff_seas = data_diff.diff(12)
data_diff_seas = data_diff_seas.dropna()
dec = sm.tsa.seasonal_decompose(data_diff_seas,period = 12)
dec.plot()
plt.show()
tsplot(data_diff_seas)
model = sm.tsa.statespace.SARIMAX(data['Passengers'],order = (2,1,2),seasonal_order = (1,1,2,12))
results = model.fit()
print(results.summary())
data['FORECAST'] = results.predict(start = 120,end = 144,dynamic = True)
data[['Passengers','FORECAST']].plot(figsize = (12,8))
exp = [data.iloc[i,0] for i in range(120,len(data))]
pred = [data.iloc[i,1] for i in range(120,len(data))]
data = data.drop(columns = 'FORECAST')
print(mean_absolute_error(exp,pred))
from pandas.tseries.offsets import DateOffset
future_dates = [data.index[-1] + DateOffset(months = x)for x in range(0,25)]
df = pd.DataFrame(index = future_dates[1:],columns = data.columns)
forecast = pd.concat([data,df])
forecast['FORECAST'] = results.predict(start = 144,end = 168,dynamic = True)
forecast[['Passengers','FORECAST']].plot(figsize = (12,8))