# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pylab as plt
%matplotlib inline 
from matplotlib import rcParams

from matplotlib.pyplot import plot
rcParams['figure.figsize']=10,6
dataset=pd.read_csv("/kaggle/input/bgr.us.txt")

dataset=dataset.drop(['High','Low','Close','Volume','OpenInt'],axis=1)
dataset['Date'] = pd.to_datetime(dataset['Date'])

dataset

dataset['Date']=pd.to_datetime(dataset['Date'],infer_datetime_format=True)
indexed_dataset=dataset.set_index(["Date"])
from datetime import datetime
indexed_dataset.head(5)
plt.xlabel("Date")
plt.ylabel("Number of air passengers")
plt.plot(indexed_dataset)
rolmean=indexed_dataset.rolling(window=12).mean()
rolstd=indexed_dataset.rolling(window=12).std()

print(rolmean,rolstd)
orig=plt.plot(indexed_dataset,color='blue',label='Original')
mean=plt,plot(rolmean,color='red',label='rolling mean')

std=plt.plot(rolstd,color='black',label='rolling std')

plt.legend(loc='best')

plt.title('rolling mean and standard deviation')

plt.show()

from statsmodels.tsa.stattools import adfuller



print('results of dickey fuller test;')

dftest=adfuller(indexed_dataset['Open'],autolag='AIC')

dfoutput=pd.Series(dftest[0:4],index=['Test Statstic','p-value','#Lags Used','Numeber of observations Used'])

for key,value in dftest[4].items():

    dfoutput['Critical Value (%s)'%key]=value

print (dfoutput)    
indexed_dataset_logScale=np.log(indexed_dataset)

plt.plot(indexed_dataset_logScale)
movingAverage=indexed_dataset_logScale.rolling(window=12).mean()

movingSTD=indexed_dataset_logScale.rolling(window=12).std()

plt.plot(indexed_dataset_logScale)

plt.plot(movingAverage,color='red')
datasetLogScaleMinusMovingAverage=indexed_dataset_logScale-movingAverage

datasetLogScaleMinusMovingAverage.head(12)

datasetLogScaleMinusMovingAverage.dropna(inplace=True)

datasetLogScaleMinusMovingAverage.head(10)
plt.plot(datasetLogScaleMinusMovingAverage,color='red')
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):

    movingAverage=timeseries.rolling(window=12).mean()

    movingSTD=timeseries.rolling(window=12).std()



    

    orig=plt.plot(timeseries,color='blue',label='Original')

    mean=plt.plot(movingAverage,color='red',label='Rolling mean')

    std=plt.plot(movingSTD,color='black',label='rolling std')

    plt.legend(loc='best')

    plt.title('rolling mean& standard deviation')

    plt.show(block=False)

    

    print('results of dickey fuller test')

    dftest=adfuller(timeseries['Open'],autolag='AIC')

    dfoutput=pd.Series(dftest[0:4],index=['Test Stastic','p-value','#lags used','Number of observations used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)' %key]=value

    print(dfoutput)    
test_stationarity(datasetLogScaleMinusMovingAverage)
exponentialDecayWeightedAverage=indexed_dataset_logScale.ewm(halflife=12,min_periods=0,adjust=True).mean()

plt.plot(indexed_dataset_logScale)

plt.plot(exponentialDecayWeightedAverage,color='red')
datasetLogScaleMinusMovingExponentialDecayAverage=indexed_dataset_logScale-exponentialDecayWeightedAverage

test_stationarity(datasetLogScaleMinusMovingExponentialDecayAverage)
datasetLogDiffshifting=indexed_dataset_logScale-indexed_dataset_logScale.shift()

plt.plot(datasetLogDiffshifting)
#by running previous cell we get parameter of arima model D=1 because we have shifted our model once ie we ahve used .shift once 

datasetLogDiffshifting.dropna(inplace=True)

test_stationarity(datasetLogDiffshifting)
#now let us see the components of time series

from statsmodels.tsa.seasonal import seasonal_decompose

decomposition=seasonal_decompose(indexed_dataset_logScale,freq=12)



trend=decomposition.trend

seasonal =decomposition.seasonal

residual=decomposition.resid





plt.subplot(411)

plt.plot(indexed_dataset_logScale,label="Original")

plt.legend(loc='best')

plt.subplot(412)

plt.plot(trend,label='Trend')

plt.legend(loc='best')

plt.subplot(413)

plt.plot(seasonal,label='Seasonality')

plt.legend(loc='Best')

plt.subplot(414)

plt.plot(residual,label='Residuals')

plt.legend(loc='Best')

plt.tight_layout()



decomposedLogData=residual

decomposedLogData.dropna(inplace=True)

test_stationarity(decomposedLogData)

decomposedLogData=residual

decomposedLogData.dropna(inplace=True)

test_stationarity(decomposedLogData)
from statsmodels.tsa.stattools import acf,pacf



lag_acf=acf(datasetLogDiffshifting, nlags=20)

lag_pacf=pacf(datasetLogDiffshifting,nlags=20,method='ols') #ols is ordinary least square method



plt.subplot(121)

plt.plot(lag_acf)

plt.axhline(y=0,linestyle='-',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffshifting)),linestyle='-',color='gray')

plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffshifting)),linestyle='-',color='gray')

plt.title('Autocorrelation Function')



#plot PACF

plt.subplot(122)

plt.plot(lag_pacf)

plt.axhline(y=0,linestyle='-',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffshifting)),linestyle='-',color='gray')

plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffshifting)),linestyle='-',color='gray')

plt.title('partial Autocorrelation function')

plt.tight_layout()
#we have plotted acf and pacf graph in order to calculate p and q we see where acf and pacf graph cuts 0 for the first time . pacf gives p and acf graph gives q.

# here p =2 and q=2



from statsmodels.tsa.arima_model import ARIMA

#AR model

model=ARIMA(indexed_dataset_logScale,order=(1,1,0))

results_AR=model.fit(disp=-1)

plt.plot(datasetLogDiffshifting)

plt.plot(results_AR.fittedvalues,color='red')

plt.title('RSS:%.4f'% sum((results_AR.fittedvalues-datasetLogDiffshifting["Open"])**2))

print('Plotting AR model')
#MA model

model=ARIMA(indexed_dataset_logScale,order=(0,1,1))

results_MA=model.fit(disp=-1)

plt.plot(datasetLogDiffshifting)

plt.plot(results_MA.fittedvalues,color='red')

plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-datasetLogDiffshifting["Open"])**2))

print('Plotting MA MODEL')
model=ARIMA(indexed_dataset_logScale,order=(1,1,1))

results_ARIMA=model.fit(disp=-1)

plt.plot(datasetLogDiffshifting)

plt.plot(results_ARIMA.fittedvalues,color='red')

plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-datasetLogDiffshifting["Open"])**2))

print('Plotting  MODEL')
predictions_ARIMA_diff=pd.Series(results_ARIMA.fittedvalues,copy=True)

print(predictions_ARIMA_diff.head())
predictions_ARIMA_diff_cumsum=predictions_ARIMA_diff.cumsum()

print(predictions_ARIMA_diff_cumsum.head())
predictions_ARIMA_log=pd.Series(predictions_ARIMA_diff_cumsum,index=indexed_dataset_logScale.index)

predictions_ARIMA_log.head()
predictions_ARIMA_log=pd.Series(indexed_dataset_logScale['Open'].ix[0],index=indexed_dataset_logScale.index)

b=pd.Series(predictions_ARIMA_diff_cumsum,index=indexed_dataset_logScale.index)



c=predictions_ARIMA_log.add(b,fill_value=0)

c.head()

predictions_ARIMA=np.exp(c)

plt.plot(indexed_dataset)

plt.plot(predictions_ARIMA)
#how to make predictions 

indexed_dataset_logScale
results_ARIMA.plot_predict(2500,3800)

results=results_ARIMA.forecast(steps=1)

results
from sklearn.metrics import mean_absolute_error as MAE