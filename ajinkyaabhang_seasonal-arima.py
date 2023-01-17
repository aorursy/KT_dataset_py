from datetime import datetime

import numpy as np             #for numerical computations like log,exp,sqrt etc

import pandas as pd            #for reading & storing data, pre-processing

import matplotlib.pylab as plt #for visualization

#for making sure matplotlib plots are generated in Jupyter notebook itself

%matplotlib inline             

from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.stattools import acf, pacf

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.arima_model import ARIMA

from matplotlib.pylab import rcParams
df=pd.read_csv('../input/air-passengers/AirPassengers.csv')
df.head()
df.shape
df.isnull().sum()
df.isna().sum()
# Convert Month into Datetime

df['Month']=pd.to_datetime(df['Month'])
df.head()
df.set_index('Month',inplace=True)
df.head()
df.describe()
df.plot()
### Testing For Stationarity



from statsmodels.tsa.stattools import adfuller
#Ho: It is non stationary

#H1: It is stationary



def adfuller_test(sales):

    result=adfuller(sales)

    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']

    for value,label in zip(result,labels):

        print(label+' : '+str(value) )

    if result[1] <= 0.05:

        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")

    else:

        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
adfuller_test(df['#Passengers'])
#Determine rolling statistics

rolmean = df.rolling(window=12).mean() #window size 12 denotes 12 months, giving rolling mean at yearly level

rolstd = df.rolling(window=12).std()
#Plot rolling statistics

orig = plt.plot(df, color='blue', label='Original')

mean = plt.plot(rolmean, color='red', label='Rolling Mean')

plt.legend(loc='best')

plt.title('Rolling Mean & Standard Deviation')

plt.show(block=False)
#Estimating trend

df_logScale = np.log(df)

plt.plot(df_logScale)
#The below transformation is required to make series stationary

movingAverage = df_logScale.rolling(window=12).mean()

movingSTD = df_logScale.rolling(window=12).std()

plt.plot(df_logScale)

plt.plot(movingAverage, color='red')
df_logScale_ma = df_logScale - movingAverage

df_logScale_ma.head(12)
#Remove NAN values

df_logScale_ma.dropna(inplace=True)

df_logScale_ma.head(10)
## Again test dickey fuller test

adfuller_test(df_logScale_ma)
df_logScale_ma.plot()
df_logScale_shift = df_logScale - df_logScale.shift()

plt.plot(df_logScale_shift)
df_logScale_shift.dropna(inplace=True)

df_logScale_shift.plot()
## Again test dickey fuller test

adfuller_test(df_logScale_shift)
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm



fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(211)

fig = sm.graphics.tsa.plot_acf(df_logScale_shift.values.squeeze(),lags=40,ax=ax1)

ax2 = fig.add_subplot(212)

fig = sm.graphics.tsa.plot_pacf(df_logScale_shift,lags=40,ax=ax2)
model=ARIMA(df,order=(2,1,2))

model_fit=model.fit()
model_fit.summary()
model_fit.aic
df['forecast']=model_fit.predict(start=90,end=103,dynamic=True)

df[['#Passengers','forecast']].plot(figsize=(12,8))
model=sm.tsa.statespace.SARIMAX(df['#Passengers'],order=(2,1,2),seasonal_order=(2,1,2,12))

results=model.fit()
results.aic
results.summary()
df['forecast']=results.predict(start=90,end=800,dynamic=True)

df[['#Passengers','forecast']].plot(figsize=(12,8))