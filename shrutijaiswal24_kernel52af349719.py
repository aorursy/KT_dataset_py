import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn import metrics
from math import sqrt
import matplotlib.pyplot as plt



data=pd.read_csv("../input/train_csv.csv")
data=data.set_index(['time'])
data.index = pd.to_datetime(data.index)
data=DataFrame(data)
data=data.drop('id',axis=1)
data.tail(10)

data.info()
data.describe()
data.isna().sum() # checking the null values
######################Data Visualization###########################

data.feature.plot(legend=True,figsize=(15,4))
plt.grid(True)


mean=data.rolling(window=3).mean()
std=data.rolling(window=3).std()
print('mean',mean,'\n' 'SD',std)
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 4
orig = plt.plot(data, color='blue',label='Original')

mean = plt.plot(mean, color='red', label='Rolling Mean')
std = plt.plot(std, color='black', label = 'Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show(block=False)


# lets check whether the train is sationary ts or non stationary
from statsmodels.tsa.stattools import adfuller
result = adfuller(data['feature'],autolag='AIC')
print('ADF statstic : %f'%result[0])
print('p value : %f'%result[1])
print('critical values')
for key,value in result[4].items():
    print(key ,value)
# Clearly  , from Dickey - Fuller test we get p value approx equal to zero which means series is stationary but from 
#the above graph we see that mean and std is not constant and its changing with time.
#So why do ADF test indicate stationarity ? 
#Because the Dickey-Fuller (ADF) test is tailored for detecting nonstationarity in the form of a unit root in the process.
# (The test equations explicitly allow for a unit root) 
#However, they are not tailored for detecting other forms of nonstationarity. 
#Therefore, it is not surprising that they do not detect nonstationarity of the seasonal kind.

#For better visiualisation lets decompose the timeseries###########
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(np.asarray(data['feature'] ), model = 'additive',freq=10)

# I have set freq=10 because we have the data at every interval of 10 sec
decomposition.plot()
plt.show()
# Now before modelling ,we have to make this series stationary and there are several ways to make series stationary : 
#differencing the time series, or making the values smaller by taking log transform
# we will use log transform
# lets create a function for rolling statistics and their plot :
def test_stationarity(timeseries):   
    #Determining rolling statistics
    rolmean = timeseries.rolling(window = 3).mean()
    rolstd = timeseries.rolling(window = 3).std()
    
    #plotting rolling statistics
    orig = plt.plot(timeseries, color = 'blue', label = 'Original')
    mean = plt.plot(rolmean, color = 'red', label = 'Rolling Mean')
    std = plt.plot(rolstd, color = 'black', label = 'Rolling Std')
    plt.legend(loc = 'best')
    plt.title('Rolling  Mean & Standard Deviation')
    plt.show()
    
    
ts=data
ts_log1=np.log(ts)
plt.plot(ts_log1)
ts_log1
ts_log_diff1 = ts_log1 - ts_log1.shift()
ts_log_diff1.dropna()
test_stationarity(ts_log_diff1)
# we are getting better than before , lets do once again
ts_log2=np.log(ts_log1)
ts_log_diff2 = ts_log2 - ts_log2.shift()
ts_log_diff2.dropna(inplace=True)
test_stationarity(ts_log_diff2)

ts_log2.head(10)
#########Time series modelling######
######## we can use AR , MA or ARIMA model and for that we have to find p,q values using ACF and PACF

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
plot_acf(ts_log_diff2) # to find the value of q
plt.show()
plot_pacf(ts_log_diff2)# to find the value of p
plt.show()
# p and q are the biggest lag after which other lags are not significant in the PACF and ACF graph resp.
#So clearly from graph , p=1,2,3 and q=1,2
from statsmodels.tsa.arima_model import ARIMA
#######AR Model######################
model=ARIMA(ts_log2,order=(2,1,0))## q=0 giving AR model
model_fit=model.fit(disp=False)
plt.plot(ts_log_diff2,label='data')
plt.plot(model_fit.fittedvalues,label='AR_predication')
plt.legend(loc = 'best')
plt.title('AR Modelling ')
plt.show()
AR_pred=model_fit.fittedvalues

####### MA Model##############
model=ARIMA(ts_log2,order=(0,1,2))
model_fit=model.fit(disp=False)
plt.plot(ts_log_diff2,label='data')
plt.plot(model_fit.fittedvalues,label='MA_predication')
plt.legend(loc = 'best')
plt.title('MA Modelling ')
plt.show()
MA_pred=model_fit.fittedvalues
# Combining =ARIMA model
model=ARIMA(ts_log2,order=(2,1,2))
model_fit=model.fit(disp=False)
plt.plot(ts_log_diff2,label='data')
plt.plot(model_fit.fittedvalues,label='ARIMA_predication')
plt.legend(loc = 'best')
plt.title('ARIMA Modelling ')
plt.show()
ARIMA_pred=model_fit.fittedvalues

err_AR=sqrt(metrics.mean_squared_error(ts_log_diff2.feature,AR_pred))
err_MA=sqrt(metrics.mean_squared_error(ts_log_diff2.feature,MA_pred))
err_ARIMA=sqrt(metrics.mean_squared_error(ts_log_diff2.feature,ARIMA_pred))

print('RMSE for AR',err_AR)

print('RMSE for MA',err_MA)

print('RMSE for ARIMA',err_ARIMA)
#RMSE for ARIMA is compartively less than other two , so we will go for ARIMA Modelling
ARIMA_pred1=pd.Series(ARIMA_pred,copy=True)
ARIMA_pred1.head()
#Clearly the first time index is missing because  we took a lag by 1 and first element doesn’t have anything before it to 
#subtract from.So we will first determine the cumulative sum and add it to the base number.
ARIMA_pred2= ARIMA_pred1.cumsum()
ARIMA_pred2.head()
cumsum_series = pd.Series(ts_log2.feature.ix[0], index=ts_log2.index)
cumsum_series= cumsum_series.add(ARIMA_pred2,fill_value=0)
ARIMA_pred3=cumsum_series
ARIMA_pred3.head(10)
# Now we will take exponent  and compare with original values:
ARIMA_pred4=np.exp( np.exp(ARIMA_pred3))# because we have taken two log transform
ARIMA_pred4

plt.plot(data)
plt.plot(ARIMA_pred4)
error=sqrt(metrics.mean_squared_error(data.feature,ARIMA_pred4))
plt.title('RMSE: %.4f'% error)
##forcasting the values by ARIMA Model#######
model_fit.plot_predict(1,120)
x=model_fit.forecast(40)#steps =40 because we have len(test_data)=40
x


############## Thanks ######################################






















