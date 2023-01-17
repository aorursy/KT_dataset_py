import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pandas_datareader.data as web
from datetime import datetime
%matplotlib inline

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
end = datetime.now()
start = datetime(end.year-2, end.month, end.day)
YES = web.DataReader("YESBANK.NS", 'yahoo', start, end)
VODAIDEA = web.DataReader("IDEA.NS", 'yahoo', start, end)
ICICI = web.DataReader("ICICIBANK.NS", 'yahoo', start, end)
SAIL = web.DataReader("SAIL.NS", 'yahoo', start, end)
REL = web.DataReader("RCOM.NS", 'yahoo', start, end)
df = pd.DataFrame({'YES': YES['Adj Close'], 'VODAFONE': VODAIDEA['Adj Close'],
                   'ICICI': ICICI['Adj Close'], 'SAIL': SAIL['Adj Close'],
                   'RELIANCE': REL['Adj Close']})
df.head()
df.plot(figsize=(10,5))
plt.legend(loc='upper right')
start = df.apply(lambda x: x / x[0])
start.plot(figsize=(10,5)).axhline(1, lw=1, color='black')
def DF(series):
    
    series = series.dropna()

    print('Dickey Fuller Test')
    test = adfuller(series, autolag='AIC')

    output = pd.Series(test[0:2], index=['Test Statistic','p-value'])
    
    for key,value in test[4].items():
        output['Critical Value (%s)'%key] = value

    print(output)
    print()
    
    if test[1] <= 0.05:
        print("The Time-Series is Stationary\n")
    else:
        print("The Time-Series is Not Stationary\n")
    
plt.figure(figsize=(10,5))
df['RELIANCE'].plot()
def trendtransformation(series):
    
    #Log transformation of the values
    LogTransformation = series.apply(lambda x : np.log(x))
    LogTransformation = LogTransformation.dropna()
    
    #Transformation of values to the power of 0.5
    PowerTransformation = series.apply(lambda x : x ** 0.5)
    PowerTransformation = PowerTransformation.dropna()
    
    #Obtaining the difference between the series and its rolling window values (i.e: series 1st value - series 12th value)  
    rollingmean = series.rolling(window = 12).mean()
    detrend = series - rollingmean
    detrend = detrend.dropna()
    

    #Plotting the transformed values of the series
    plt.figure(figsize=(15,5))

    plt.subplot(2,2,1)
    series.plot()
    plt.title("Original Value")

    plt.subplot(2,2,2)
    LogTransformation.plot()
    plt.title("Log Transformation Value")

    plt.subplot(2,2,3)
    PowerTransformation.plot()
    plt.title("Power Transformation Value")

    plt.subplot(2,2,4)
    detrend.plot()
    plt.title("Rolling Mean Value")
    
    plt.tight_layout(pad=0.5)
    plt.show(block=False)
    
    
    print("***** Log Transformation *****")
    DF(LogTransformation)
    print("***** Power Transformation *****")
    DF(PowerTransformation)
    print("***** Rolling Mean *****")
    DF(detrend)
trendtransformation(df['RELIANCE'])
rollingmean = df['RELIANCE'].rolling(window = 12).mean()
reliance = df['RELIANCE'] - rollingmean
reliance = reliance.dropna()
plt.figure(figsize=(10,5))
df['ICICI'].plot()
def randomtransformation(series):
    
    #Difference between the log transformation and its values that has been shifted by 1 place (row)
    LogTransformation = series.apply(lambda x : np.log(x))
    DifferenceOfLogTransformation = LogTransformation - LogTransformation.shift()
    DifferenceOfLogTransformation = DifferenceOfLogTransformation.dropna()

    #Difference between the power transformation and its values that has been shifted by 1  place (row)
    PowerTransformation = series.apply(lambda x : x ** 0.5)
    DifferenceOfPowerTransformation = PowerTransformation - PowerTransformation.shift()
    DifferenceOfPowerTransformation = DifferenceOfPowerTransformation.dropna()
    
        
    #Difference between the rolling mean and its values that has been shifted by 1  place (row)
    rollingmean = series.rolling(window = 12).mean()
    DifferenceOfRollTransformation = rollingmean - rollingmean.shift()
    DifferenceOfRollTransformation = DifferenceOfRollTransformation.dropna()
 

    plt.figure(figsize=(15,5))

    plt.subplot(2,2,1)
    series.plot()
    plt.title("Original Value")

    plt.subplot(2,2,2)
    DifferenceOfLogTransformation.plot()
    plt.title("Log Transformation Value")

    plt.subplot(2,2,3)
    DifferenceOfPowerTransformation.plot()
    plt.title("Power Transformation Value")

    plt.subplot(2,2,4)
    DifferenceOfRollTransformation.plot()
    plt.title("Rolling Mean Value")
    
    plt.tight_layout(pad=0.5)
    plt.show(block=False)
    

    print("***** Log Transformation *****")
    DF(DifferenceOfLogTransformation)
    print("***** Power Transformation *****")
    DF(DifferenceOfPowerTransformation)
    print("***** Rolling Mean *****")
    DF(DifferenceOfRollTransformation)

randomtransformation(df['ICICI'])
LogTransformation = df['ICICI'].apply(lambda x : np.log(x))
icici = LogTransformation - LogTransformation.shift()
icici = icici.dropna()
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARIMA
plt.figure(figsize=(12,8))

plot_acf(reliance,lags=40)
plot_pacf(reliance,lags=40)
plt.show()
model=ARIMA(reliance,order=(1,0,1))
model_fit=model.fit()
relianceforecast=model_fit.predict()
reliance.plot()
relianceforecast.plot()
plt.figure(figsize=(12,8))

plot_acf(icici,lags=40)
plot_pacf(icici,lags=40)
plt.show()
model=ARIMA(icici,order=(1,1,1))
model_fit=model.fit()
iciciforecast=model_fit.predict()
icici.plot()
iciciforecast.plot()