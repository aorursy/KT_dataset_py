import pandas as pd 

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import datetime
a=datetime.datetime.strptime('2019-10-08', '%Y-%m-%d')

print(a)

print(type(a))
def parser(x):

    return datetime.datetime.strptime(x, '%Y-%m')
data= pd.read_csv("../input/AirPassengers.csv", date_parser=parser, parse_dates=[0], index_col=[0])
data.head()
#plotting the data

plt.plot(data['#Passengers'])
#determining the rolling stats of data

#Rolmean is used to smoothen the graph. Weactually deal with noice using rolling mean.

rolmean= data.rolling(window=12).mean()

rolstd= data.rolling(window=12).std()



orig= plt.plot(data, color='blue', label='original')

rolmean= plt.plot(rolmean, color='red', label='rolmean')

rolstd= plt.plot(rolstd, color='black', label='rolstd')

plt.legend(loc='best')

plt.title('Plot for original data')

plt.show()

#performing the dickey fuller test on original data

from statsmodels.tsa.stattools import adfuller

org_test= adfuller(data['#Passengers'])

org_output=pd.Series(org_test[0:4], index=['TestStatistics', 'P-value', '#lagsUsed', '#obs used'])

for key, value in org_test[4].items():

    org_output['critical_value {}'.format(key)]=value

org_output

#the stats came out to be non stationary



#The Time Series is still not stationary here 

#Trend – varying mean over time. For eg, in this case we saw that on average, the number of passengers was growing over time.

#Seasonality – variations at specific time-frames. 
#using log of data and check stationarity

data_log_value= np.log(data)

mov_avg= data_log_value.rolling(window=12).mean()

mov_std= data_log_value.rolling(window=12).std()



plt.plot(data_log_value, label='data_log_value')

plt.plot(mov_avg, color='red', label='mov_avg')

plt.plot(mov_std, color='green', label='mov_std')

plt.legend()



#still the plot is not stationary



#performing the dickey fuller test on original data

from statsmodels.tsa.stattools import adfuller

org_test= adfuller(data_log_value['#Passengers'])

org_output=pd.Series(org_test[0:4], index=['TestStatistics', 'P-value', '#lagsUsed', '#obs used'])

for key, value in org_test[4].items():

    org_output['critical_value {}'.format(key)]=value

org_output
#Lets define a function to check stationarity further

def check_stationarity(dataset):

    rolling_mean= dataset.rolling(window=12).mean()

    rolling_std= dataset.rolling(window=12).std()

    plt.plot(dataset, color='blue', label='original')

    plt.plot(rolling_mean, color='red', label='rolling_mean')

    plt.plot(rolling_std, color='green', label='rolling_std')

    plt.legend(loc='best')

    plt.show()

    

    from statsmodels.tsa.stattools import adfuller

    test= adfuller(dataset['#Passengers'], autolag='AIC')

    output=pd.Series(test[0:4], index=['TestStatistics', 'P-value', '#lagsUsed','#ObsUsed'])

    for key,value in test[4].items():

        output['Critical_value {}'.format(key)]= value

   

    display(output)  
#lets shift the data by one and check the stationarity for data without log

data_1Lag= data.shift(periods=1)

dataMinusdata1Lag= data-data_1Lag

dataMinusdata1Lag.dropna(inplace=True)
check_stationarity(dataMinusdata1Lag)



#seems like the rolling mean is constant but still variance is there so it cant be stationary.
#1 Checking stationarity for #data_log_valueMinusmov_avg 

#It is taking rolling mean(12)of log data and then subratcting from original log data



data_log_value= np.log(data)

mov_avg= data_log_value.rolling(window=12).mean()

mov_std= data_log_value.rolling(window=12).std()



data_log_valueMinusmov_avg= data_log_value-mov_avg

data_log_valueMinusmov_avg.dropna(inplace=True)



check_stationarity(data_log_valueMinusmov_avg)



#Here the result came out is stationary data
#2 (Without subtraction)

#Exponential weighted average

exponentialDecayWeightedAvg= data_log_value.ewm(halflife=12, min_periods=0, adjust=True).mean()

plt.plot(data_log_value)

plt.plot(exponentialDecayWeightedAvg, color='red')



#The Exponential weighted average data is not stationary
#Now taking the subtracted data for Exponential weighted average



data_log_valueMinusexponentialDecayWeightedAvg= data_log_value-exponentialDecayWeightedAvg

check_stationarity(data_log_valueMinusexponentialDecayWeightedAvg)



#The data comes out here is stationary 
#3 Differencing

#Shifting Log value by one

data_log_value_shift= data_log_value-data_log_value.shift(1)

data_log_value_shift.dropna(inplace=True)

plt.plot(data_log_value_shift)
check_stationarity(data_log_value_shift)

#the data comes out here is also stationary.



#Lets take this to build our ARIMA model
#Components of Timeseries

#Decomposition



from statsmodels.tsa.seasonal import seasonal_decompose

decompose= seasonal_decompose(data_log_value)
trend= decompose.trend

seasonal= decompose.seasonal

residual=decompose.resid



plt.subplot(411)

plt.plot(data_log_value, label='original')

plt.legend()



plt.subplot(412)

plt.plot(trend, color='red', label='trend')

plt.legend()



plt.subplot(413)

plt.plot(seasonal, color='green', label='seasonality')

plt.legend()



plt.subplot(414)

plt.plot(residual, color='red', label='residual')

plt.legend()

residual.dropna(inplace=True)

check_stationarity(residual)
#Lets build our model using differencing as it is very popular technique. Apart from that its easier to add noice and seasonality

#back into predicted value
#ACF and #PACF value

from statsmodels.tsa.stattools import acf, pacf

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



acf= acf(data_log_value_shift, nlags=10)

pacf=pacf(data_log_value_shift, nlags=10, method='ols')
#Plot ACF

plt.figure(figsize=(10,6))

plt.subplot(121)

plt.plot(acf)

plt.axhline(y=0, linestyle='--', color='black')

plt.axhline(y=-1.96/np.sqrt(len(data_log_value_shift)), linestyle='--', color='black')

plt.axhline(y=1.96/np.sqrt(len(data_log_value_shift)), linestyle='--', color='black')

plt.axvline(x=0, linestyle='--', color='grey')

plt.axvline(x=1, linestyle='--', color='grey')

plt.axvline(x=2, linestyle='--', color='grey')

plt.axvline(x=3, linestyle='--', color='grey')

plt.axvline(x=4, linestyle='--', color='grey')

plt.title('AutoCorrelation Function')





plt.subplot(122)

plt.plot(pacf)

plt.axhline(y=0, linestyle='--', color='black')

plt.axhline(y=-1.96/np.sqrt(len(data_log_value_shift)), linestyle='--', color='black')

plt.axhline(y=1.96/np.sqrt(len(data_log_value_shift)), linestyle='--', color='black')

plt.axvline(x=0, linestyle='--', color='grey')

plt.axvline(x=1, linestyle='--', color='grey')

plt.axvline(x=2, linestyle='--', color='grey')

plt.axvline(x=3, linestyle='--', color='grey')

plt.title('PartialAutoCorrelation Function')
plot_acf(data_log_value_shift, lags=10)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(data_log_value_shift)),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(data_log_value_shift)),linestyle='--',color='gray')

plt.title('Autocorrelation Function')



plot_pacf(data_log_value_shift, lags=10)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(data_log_value_shift)),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(data_log_value_shift)),linestyle='--',color='gray')

plt.title('PartialAutocorrelation Function')
#we can take (1,1) or (2,2) value as both lies above the boundary line(dashed line)

#ARIMA Model

import warnings

warnings.filterwarnings('ignore')



from statsmodels.tsa.arima_model import ARIMA



model_ARIMA1= ARIMA(data_log_value, order=(2,1,2))

result1= model_ARIMA1.fit()

print(result1.aic)



plt.plot(data_log_value_shift, color='blue', alpha=0.5 )

plt.plot(result1.fittedvalues, color='red')





from sklearn.metrics import mean_squared_error

print(np.sqrt(mean_squared_error(data_log_value_shift, result1.fittedvalues)))

print('RMSE value is {}'.format(sum((data_log_value_shift['#Passengers']-result1.fittedvalues)**2)))
#Lets use AIC to find the best values

import itertools

p=d=q=range(0,5)

pdq= list(itertools.product(p,d,q))



for i in pdq:

    try:

         model_ARIMA1= ARIMA(data_log_value, order=i)

         result1= model_ARIMA1.fit()

         print(i, result1.aic)

    except:

        continue
predicted_ARIMA_diff= result1.fittedvalues

predicted_ARIMA_diff.head()
predicted_ARIMA_diff_cumsum= np.cumsum(predicted_ARIMA_diff)

predicted_ARIMA_diff_cumsum.head(10)
predicted_log_value= pd.Series(data_log_value['#Passengers'].iloc[0], index=data_log_value.index).add(predicted_ARIMA_diff_cumsum

                                                                                                   ,fill_value=0)
predicted_log_value.head()
prediction_ARIMA= np.exp(predicted_log_value)

plt.plot(data)

plt.plot(prediction_ARIMA, color='red', alpha=0.7)

plt.show()



from sklearn import metrics

metrics.r2_score(data, prediction_ARIMA)
data.shape
#lets predict for next 10 years

result1.plot_predict(start='1953-07-01', end='1962-12-01')
#forcasting the log value for next 10 years

result1.forecast(120)