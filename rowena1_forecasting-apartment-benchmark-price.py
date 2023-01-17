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
import pandas as pd

df=pd.read_csv('/kaggle/input/apartment-benchmark-prices/Dataset.csv',index_col=0,header=0,squeeze=True)

validation=pd.read_csv('/kaggle/input/apartment-benchmark-prices/Validation.csv',index_col=0,header=0,squeeze=True)

print('Dataset %d, Validation %d' %(len(df),len(validation)))
import matplotlib.pyplot as plt

df.plot()
plt.figure()

plt.subplot(211)

df.hist()

plt.subplot(212)

df.plot(kind='kde')

plt.show()
#Check stationarity

from statsmodels.tsa.stattools import adfuller



result=adfuller(df.diff().dropna())

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])

print('Critical Values:')

for key, value in result[4].items():

    print('\t%s: %.4f' % (key,value)) 
#Plot ACF and PACF for first order differencing



from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import matplotlib.pyplot as plt 

import numpy as np



plt.figure()

plt.subplot(211)

plot_acf(df.diff().dropna(),ax=plt.gca())

plt.subplot(212)

plot_pacf(df.diff().dropna(),ax=plt.gca())

plt.show()
# Check second order differencing



result=adfuller(df.diff().diff().dropna())

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])

print('Critical Values:')

for key, value in result[4].items():

    print('\t%s: %.4f' % (key,value)) 
#Plot ACF and PACF for second order differencing



from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import matplotlib.pyplot as plt 

import numpy as np



plt.figure()

plt.subplot(211)

plot_acf(df.diff().diff().dropna(),ax=plt.gca())

plt.subplot(212)

plot_pacf(df.diff().diff().dropna(),ax=plt.gca())

plt.show()
# Manually configure ARIMA



from sklearn.metrics import mean_squared_error

from statsmodels.tsa.arima_model import ARIMA

from math import sqrt

import numpy as np



# Calculate mean absolute percentage error

def mean_abs_percent_error(obs,yhat):

    obs, yhat=np.array(obs),np.array(yhat)

    return np.mean(np.abs((obs-yhat)/obs))*100



# Prepare data

X=df.values

X=X.astype('float32')

train_size=int(len(X)*0.7)

train, test=X[0:train_size], X[train_size:]



# Walk-forward validation

history=[x for x in train]

predictions=list()

for i in range(len(test)):

    #predict

    model=ARIMA(history,order=(1,2,1))

    model_fit=model.fit(disp=0)

    yhat=model_fit.forecast()[0]

    predictions.append(yhat)

    #observation

    obs=test[i]

    history.append(obs)

    print('Predicted= %.4f, Expected= %.4f' % (yhat,obs))



# Report performance

rmse=sqrt(mean_squared_error(test,predictions))

print('RMSE: %.4f' % (rmse))
#plot residual errors for ARIMA model

from pandas import DataFrame

#errors

residuals=[test[i]-predictions[i] for i in range(len(test))]

residuals=DataFrame(residuals)

plt.figure()

plt.subplot(211)

residuals.hist(ax=plt.gca())

plt.subplot(212)

residuals.plot(kind='kde',ax=plt.gca())

plt.show()
residuals.describe()
print(model_fit.summary())
#summarize residual errors from bias corrected forecasts

#walk-forward validation

history=[x for x in train]

predictions=list()

bias=216

for i in range(len(test)):

    #predict

    model=ARIMA(history,order=(1,2,1))

    model_fit=model.fit(disp=0)

    yhat=bias+float(model_fit.forecast()[0])

    predictions.append(yhat)

    #observation

    obs=test[i]

    history.append(obs)

# Report performance

rmse=sqrt(mean_squared_error(test,predictions))

mape=mean_abs_percent_error(test,predictions)

print('RMSE: %.4f, Mean Absolute Percentage Error: %.4f' %(rmse,mape))



#Summarize residual errors

residuals=[test[i]-predictions[i] for i in range(len(test))]

residuals=DataFrame(residuals)

print('')

print('Residuals:')

print(residuals.describe())



#Plot residual errors

plt.figure()

plt.subplot(211)

residuals.hist(ax=plt.gca())

plt.subplot(212)

residuals.plot(kind='kde',ax=plt.gca())

plt.show()
#Evaluate finalized model on validation dataset

from pandas import read_csv

import matplotlib.pyplot as plt

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.arima_model import ARIMAResults

from sklearn.metrics import mean_squared_error

from math import sqrt

import numpy as np



#Load and prepare datasets

df=read_csv('/kaggle/input/apartment-benchmark-prices/Dataset.csv',index_col=0,header=0,squeeze=True)

X=df.values.astype('float32')

history=[x for x in X]

Validation=read_csv('/kaggle/input/apartment-benchmark-prices/Validation.csv',index_col=0,header=0,squeeze=True)

y=Validation.values.astype('float32')

model=ARIMA(X,order=(1,2,1))

model_fit=model.fit(disp=0)

bias=216

#Make first prediction

predictions=list()

yhat=bias+float(model_fit.forecast()[0])

predictions.append(y[0])

print('Predicted= %.4f, Expected= %.4f' % (yhat,y[0]))



#Rolling forecasts

for i in range(1,len(y)):

    #predict

    model=ARIMA(history,order=(1,2,0))

    model_fit=model.fit(disp=0)

    yhat=bias+float(model_fit.forecast()[0])

    predictions.append(yhat)

    #Observation

    obs=y[i]

    history.append(obs)

    print('Predicted= %.4f, Expected= %.4f' % (yhat,obs))

    

#Report performance

rmse=sqrt(mean_squared_error(y,predictions))

mape=mean_abs_percent_error(y,predictions)

print('')

print('RMSE: %.4f, Mean Absolute Percentage Error: %.4f' % (rmse,mape))

plt.plot(y)

plt.plot(predictions,color='red')

plt.show()

#Let's try the model's prediction power for 12-steps

#Load and prep data

import numpy as np

from datetime import datetime, timedelta



df=pd.read_csv('/kaggle/input/apartment-benchmark-prices/Dataset.csv',header=0,squeeze=True)

df.drop('Date',axis=1,inplace=True)

index=pd.date_range(start='2005',periods=153,freq='M')

df.index=index

X=df['Apartment']



Validation=pd.read_csv('/kaggle/input/apartment-benchmark-prices/Validation.csv',header=0,squeeze=True)

Validation.drop('Date',axis=1,inplace=True)

index2=pd.date_range(start='2017-10',periods=12,freq='M')

Validation.index=index2

y=Validation['Apartment']



#Direct forecast

mod1=ARIMA(X,order=(1,2,1),freq='M')

model_fit=mod1.fit()

fc,se,conf=model_fit.forecast(12,alpha=0.05)



#Turn into pandas Series

fc_series=pd.Series(fc,index=Validation.index)

lower_series=pd.Series(conf[:,0],index=Validation.index)

upper_series=pd.Series(conf[:,1],index=Validation.index)



#Get residuals

resid1=[y[i]-fc_series[i] for i in range(len(y))]





#Report performance

rmse=sqrt(mean_squared_error(y,fc_series))

mape=mean_abs_percent_error(y,fc_series)

print('RMSE: %.4f, Mean Absolute Percentage Error: %.4f' % (rmse,mape))



#Plot graph

plt.plot(y,label='Expected')

plt.plot(fc_series,label='Predicted',color='red')

plt.fill_between(lower_series.index,lower_series,upper_series,

                color='k',alpha=.15)

plt.show()



#How would an exponential smoothing model perform

#in comparison?



from statsmodels.tsa.holtwinters import ExponentialSmoothing



df=pd.read_csv('/kaggle/input/apartment-benchmark-prices/Dataset.csv',header=0,squeeze=True)

df.drop('Date',axis=1,inplace=True)

index=pd.date_range(start='2005',periods=153,freq='M')

df.index=index

X=df['Apartment']



#Split into train & test sets

X=X.astype('float32')

train_size=int(len(X)*0.7)

train, test=X[0:train_size], X[train_size:]



# Walk-forward validation

history=[x for x in train]

predictions=list()

for i in range(len(test)):

    #predict

    model=ExponentialSmoothing(history,trend='add',damped=True)

    model_fit=model.fit(remove_bias=True)

    yhat=model_fit.forecast()[0]

    predictions.append(yhat)

    #observation

    obs=test[i]

    history.append(obs)

    print('Predicted= %.4f, Expected= %.4f' % (yhat,obs))



# Report performance

rmse=sqrt(mean_squared_error(test,predictions))

print('RMSE: %.4f' % (rmse))

#Let's try the exponential smoothing 

#model's prediction power for 12-steps

#Load and prep data

import numpy as np

from datetime import datetime, timedelta



df=pd.read_csv('/kaggle/input/apartment-benchmark-prices/Dataset.csv',header=0,squeeze=True)

df.drop('Date',axis=1,inplace=True)

index=pd.date_range(start='2005',periods=153,freq='M')

df.index=index

X=df['Apartment']



Validation=pd.read_csv('/kaggle/input/apartment-benchmark-prices/Validation.csv',header=0,squeeze=True)

Validation.drop('Date',axis=1,inplace=True)

index2=pd.date_range(start='2017-10',periods=12,freq='M')

Validation.index=index2

y=Validation['Apartment']



#Direct forecast

mod2=ExponentialSmoothing(X,trend='add',damped=True,freq='M')

model_fit=mod2.fit(remove_bias=True)

fc=model_fit.forecast(12)



#Get residuals

resid2=[y[i]-fc[i] for i in range(len(y))]





#Report performance

rmse=sqrt(mean_squared_error(y,fc))

mape=mean_abs_percent_error(y,fc)

print('RMSE: %.4f, Mean Absolute Percentage Error: %.4f' % (rmse,mape))



#Plot graph

plt.plot(y,label='Expected')

plt.plot(fc,label='Predicted',color='red')

plt.show()

#Blending the forecasts from 

#ARIMA and ETS models.

#Weights using RMSE



Blend_fc=(0.47*fc_series+0.53*fc)



#Report performance

rmse=sqrt(mean_squared_error(y,Blend_fc))

mape=mean_abs_percent_error(y,Blend_fc)

print('RMSE: %.4f, Mean Absolute Percentage Error: %.4f' % (rmse,mape))



#Plot graph

plt.plot(y,label='Expected')

plt.plot(Blend_fc,label='Predicted')

plt.show()
#OK! Let's try and forecast for the 12 months

#from Sep 2018 and see what we get.



forecast_index=pd.date_range(start='2018-10',periods=12,freq='M')



#stick validation data to the dataset to train.

df1=df.append(Validation)

X=df1['Apartment']



#retrain mod1 and mod2 on df1

#Direct forecast

mod1=ARIMA(X,order=(1,2,1),freq='M')

model_fit=mod1.fit()

fc,se,conf=model_fit.forecast(12,alpha=0.05)



#Turn into pandas Series

fc_series=pd.Series(fc,index=forecast_index)

lower_series=pd.Series(conf[:,0],index=forecast_index)

upper_series=pd.Series(conf[:,1],index=forecast_index)



mod2=ExponentialSmoothing(X,trend='add',damped=True,freq='M')

model_fit=mod2.fit(remove_bias=True)

fc=model_fit.forecast(12)



#Blending the forecasts from 

#ARIMA and ETS models.

#Weights using RMSE



forecast=0.47*fc_series+0.53*fc
#Import actuals for Oct 2018 ~ Sep 2019

#https://www.crea.ca/housing-market-stats/mls-home-price-index/hpi-tool/

actuals=[693200,677500,675200,670200,671500,667500,667500,664200,654700,653200,654000,651500]

y=pd.Series(actuals,index=forecast_index)
#Report performance

rmse=sqrt(mean_squared_error(y,forecast))

mape=mean_abs_percent_error(y,forecast)

print('RMSE: %.4f Mean Absolute Percent Error: %.4f' % (rmse,mape))



#Plot graph

plt.plot(y,label='Expected')

plt.plot(forecast,label='Predicted',color='red')

plt.fill_between(lower_series.index,lower_series,upper_series,

                color='k',alpha=.15)

plt.show()
