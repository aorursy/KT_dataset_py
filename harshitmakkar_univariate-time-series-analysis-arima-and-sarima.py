import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np

%matplotlib inline

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = (15, 6)
df = pd.read_csv('../input/international-airline-passengers.csv')
df.head()
df = df.drop(df.index[144])
df['Month'] = pd.to_datetime(df['Month'],yearfirst=True)
df.info()
df.index = df['Month']
df = df.drop('Month',axis=1)
df.columns = ['Passengers']
df.head()
df.plot()
from statsmodels.tsa.stattools import adfuller as adf

def test_stationarity(timeseries):

    

    #Determing rolling statistics

    rolmean = timeseries.rolling(12).mean()

    rolstd = timeseries.rolling(12).std()



    #Plot rolling statistics:

    orig = plt.plot(timeseries, color='blue',label='Original')

    mean = plt.plot(rolmean, color='red', label='Rolling Mean')

    std = plt.plot(rolstd, color='black', label = 'Rolling Std')

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation')

    plt.show(block=False)

    

    
test_stationarity(df)
output = (adf(df['Passengers']))
output
dfoutput = pd.Series(output[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for key,value in output[4].items():

        dfoutput['Critical Value (%s)'%key] = value

print(dfoutput)
#estimating trend and seasonlity

from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(df)



trend = decomposition.trend

seasonal = decomposition.seasonal

residual = decomposition.resid



plt.subplot(411)

plt.plot(df, label='Original')

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
#we will first use arimax to account only for trend and then later will use sarimax to account for both trend and seasonality
#divide into train and validation set

train = df[:int(0.7*(len(df)))]

valid = df[int(0.7*(len(df))):]



#plotting the data

ax = train.plot()

valid.plot(ax=ax)
#building the model

from pmdarima.arima import auto_arima

model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True)

model.fit(train)



forecast = model.predict(n_periods=len(valid))

forecast = pd.DataFrame(forecast,index = valid.index,columns=['Prediction'])



#plot the predictions for validation set

plt.plot(train, label='Train')

plt.plot(valid, label='Valid')

plt.plot(forecast, label='Prediction')

plt.legend(loc='best')

plt.show()
#using sarimax to account for seasonality and then forecasting

#building the model

from pmdarima.arima import auto_arima

model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True,seasonal=True,m=12,D=1)

model.fit(train)



forecast = model.predict(n_periods=len(valid))

forecast = pd.DataFrame(forecast,index = valid.index,columns=['Prediction'])



#plot the predictions for validation set

plt.plot(train, label='Train')

plt.plot(valid, label='Valid')

plt.plot(forecast, label='Prediction')

plt.legend(loc='best')

plt.show()
#will add in more theory to explain better