import numpy as np 

import pandas as pd 

import numpy as np

import pandas as pd

import scipy as sp

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.pylab as plt

from matplotlib.pylab import rcParams

from datetime import datetime
data=pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')

data.head()
data1= data.groupby(['Date','Country_Region'])[['ConfirmedCases','Fatalities']].sum().reset_index()

data2= data1[data1['Country_Region']=='Bangladesh']

data2.drop(data2[data2['ConfirmedCases']==0].index,inplace=True)

data2['Newcases']=data2['ConfirmedCases']-data2['ConfirmedCases'].shift(1)

data2['New']= data2['Newcases'].rolling(window=3).mean()

data2.drop(data2[data2['Date']<='2020-03-14'].index,inplace=True)

bd_data=data2.drop(['Country_Region','ConfirmedCases','Fatalities','Newcases'], axis = 1)

bd_data.head()

bd_data['Date']= pd.to_datetime(bd_data['Date'])

data_fi=bd_data.set_index('Date')

plt.xlabel('Date')

plt.ylabel('NewCases')

plt.plot(data_fi['New'])
data_fi.dropna(inplace=True)

rolmean = data_fi['New'].rolling(window=4).mean()

rolstd = data_fi['New'].rolling(window =4).std()
from statsmodels.tsa.stattools import adfuller

print('Dicky fuller taste')

dftest = adfuller(data_fi['New'],autolag='AIC')

dfout = pd.Series(dftest[0:4],index=['Test statistics','P-value','Lags used','Number of observations'])

for key,values in dftest[4].items():

    dfout['Critical values(%s)'%key]=values

print(dfout)
train2_logscale = np.log(data_fi['New'])

train2_logscale.dropna(inplace=True)

plt.plot(train2_logscale)
movingAverage = train2_logscale.rolling(window=4).mean()

movingSTD = train2_logscale.rolling(window =4).std()

plt.plot(train2_logscale)

plt.plot(movingAverage,color='red')
tm_log_avg = train2_logscale-movingAverage

tm_log_avg.dropna(inplace=True)

def test_stationary(timeseries):

    movingAverage =timeseries.rolling(window=2).mean()

    movingSTD = timeseries.rolling(window=2).std()

    orig = plt.plot(timeseries,color='blue',label='Orginal')

    avg = plt.plot(movingAverage,color='black',label='Moving Average')

    std = plt.plot(movingSTD,color='red',label='Rollong std')

    plt.legend(loc='best')

    plt.title('Rolling mean and rolling std')

    plt.show()

    

    print('Dicky fuller taste')

    dftest = adfuller(data_fi['New'],autolag='AIC')

    dfout = pd.Series(dftest[0:4],index=['Test statistics','P-value','#Lags used','Number of observations'])

    for key,values in dftest[4].items():

        dfout['Critical values(%s)'%key]=values

        print(dfout)
test_stationary(tm_log_avg)
exponential= train2_logscale.ewm(halflife=1,min_periods=0,adjust=True).mean()

plt.plot(train2_logscale)

plt.plot(exponential, color='red')

mexponential = train2_logscale-exponential

mexponential.head(12)

test_stationary(mexponential)

datashifting = train2_logscale-train2_logscale.shift(1)

plt.plot(datashifting)
datashifting.dropna(inplace=True)

test_stationary(datashifting)
train2_logscale.dropna(inplace=True)

from statsmodels.tsa.seasonal import seasonal_decompose

import statsmodels.api as sm



decompose = seasonal_decompose(train2_logscale)



trend = decompose.trend

seasonal=decompose.seasonal

residual=decompose.resid



plt.subplot(411)

plt.plot(train2_logscale,label='original')

plt.legend(loc='best')



plt.subplot(412)

plt.plot(trend,label='trend')

plt.legend(loc='best')



plt.subplot(413)

plt.plot(seasonal,label='seasonal')

plt.legend(loc='best')

plt.subplot(414)

plt.plot(residual,label='residual')

plt.legend(loc='best')

plt.tight_layout()



decomposelog= residual

decomposelog.dropna(inplace=True)

test_stationary(decomposelog)
decomposedlogdata = residual

decomposedlogdata.dropna(inplace=True)

test_stationary(decomposedlogdata)
from statsmodels.tsa.stattools import acf, pacf



lag_acf=acf(datashifting, nlags=20)

lag_pacf=pacf(datashifting, nlags=20, method='ols')



plt.figure(figsize=(20,10))

plt.subplot(121)

plt.plot(lag_acf)

plt.axhline(y=0,linestyle='--',color='green')

plt.axhline(y=-1.96/np.sqrt(len(datashifting)),linestyle='--',color='green')

plt.axhline(y=1.96/np.sqrt(len(datashifting)),linestyle='--',color='green')

plt.title('Autocorrelation Function')



plt.subplot(122)

plt.plot(lag_pacf)

plt.axhline(y=0,linestyle='--',color='green')

plt.axhline(y=-1.96/np.sqrt(len(datashifting)),linestyle='--',color='green')

plt.axhline(y=1.96/np.sqrt(len(datashifting)),linestyle='--',color='green')

plt.title('Partial Autocorrelation Function')
from statsmodels.tsa.arima_model import ARIMA

model =ARIMA(train2_logscale,order=(2,1,1))

results_ARIMA=model.fit(disp=-1)

plt.plot(datashifting)

plt.plot(results_ARIMA.fittedvalues,color='red')

plt.title('RSS:%4F'%sum(results_ARIMA.fittedvalues - datashifting**2))

print('Plotting AR model')
from statsmodels.tsa.arima_model import ARIMA

model =ARIMA(train2_logscale,order=(2,1,0))

results_AR=model.fit(disp=-1)

plt.plot(datashifting)

plt.plot(results_AR.fittedvalues,color='red')

plt.title('RSS:%4F'%sum(results_AR.fittedvalues - datashifting**2))

print('Plotting AR model')
from statsmodels.tsa.arima_model import ARIMA

model =ARIMA(train2_logscale,order=(2,2,1))

results_MA=model.fit(disp=-1)

plt.plot(datashifting)

plt.plot(results_MA.fittedvalues,color='red')

plt.title('RSS:%4F'%sum(results_MA.fittedvalues - datashifting**2))

print('Plotting AR model')
from statsmodels.tsa.arima_model import ARIMA

model =ARIMA(train2_logscale,order=(2,0,1))

results=model.fit(disp=-1)

plt.plot(datashifting)

plt.plot(results.fittedvalues,color='red')

plt.title('RSS:%4F'%sum(results.fittedvalues - datashifting**2))

print('Plotting AR model')
predictions_ARIMA_diff= pd.Series(results_ARIMA.fittedvalues,copy=True)

print(predictions_ARIMA_diff.head())
prediction_ARIMA_log=pd.Series(train2_logscale.ix[0],index= train2_logscale.index)

prediction_ARIMA_log=prediction_ARIMA_log.add(predictions_ARIMA_diff,fill_value=0)

prediction_ARIMA_log.head()
sns.set(rc={'figure.figsize':(16, 8)})

predictions_ARIMA = np.exp(prediction_ARIMA_log)

plt.plot(data_fi)

plt.plot(predictions_ARIMA)
predictions_ARIMA_diff= pd.Series(results_AR.fittedvalues,copy=True)

prediction_ARIMA_log=pd.Series(train2_logscale.ix[0],index= train2_logscale.index)

prediction_ARIMA_log=prediction_ARIMA_log.add(predictions_ARIMA_diff,fill_value=0)

sns.set(rc={'figure.figsize':(16, 8)})

predictions_ARIMA = np.exp(prediction_ARIMA_log)

plt.plot(data_fi)

plt.plot(predictions_ARIMA)
predictions_ARIMA_diff= pd.Series(results_MA.fittedvalues,copy=True)

prediction_ARIMA_log=pd.Series(train2_logscale.ix[0],index= train2_logscale.index)

prediction_ARIMA_log=prediction_ARIMA_log.add(predictions_ARIMA_diff,fill_value=0)

sns.set(rc={'figure.figsize':(16, 8)})

predictions_ARIMA = np.exp(prediction_ARIMA_log)

plt.plot(data_fi)

plt.plot(predictions_ARIMA)
predictions_ARIMA_diff= pd.Series(results_ARIMA.fittedvalues,copy=True)

print(predictions_ARIMA_diff.head())
predictions_ARIMA_diff= pd.Series(results.fittedvalues,copy=True)

prediction_ARIMA_log=pd.Series(train2_logscale.ix[0],index= train2_logscale.index)

prediction_ARIMA_log=prediction_ARIMA_log.add(predictions_ARIMA_diff,fill_value=0)

sns.set(rc={'figure.figsize':(16, 8)})

predictions_ARIMA = np.exp(prediction_ARIMA_log)

plt.plot(data_fi)

plt.plot(predictions_ARIMA)
results.plot_predict(1,70)

x=results.forecast(steps=70)

values0=x[0][::-1]

values0=np.exp(values0)

values1=x[1]

values1=np.exp(values1)

values= np.concatenate((values1,values0)) 

values
x0=np.array(x[0])

x1=np.array(x[1])

x2=np.array(x[2])



x1=np.exp(x1)

x0=np.exp(x0)

x2=np.exp(x2)
def date_to_value(m,d):

   if m==4:

      x=d-13

   elif m==5:

       x=d+17

   elif m==6:

       x=+30+17

   else:

       x=0    

   return x    





def get_forcating(x0,x1,x2,i):

  if i==0:

    print('Out of range')

  else:  

      v=x0[i]+x1[i]+x2[i,1]+x2[i,0]

      v1=v/4

      return v1

n1=date_to_value(5,25)

print("the numeric value for the date is %d"%n1)
y=get_forcating(x0,x1,x2,n1)

print("The projected cases for  25-5-2020 is %d"%int(y))