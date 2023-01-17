import numpy as np 

import pandas as pd

import os

data_covid=pd.read_csv('../input/covid19-in-india/covid_19_india.csv')

import matplotlib.pyplot as plt

from datetime import datetime

import datetime as dt

from datetime import date

from statsmodels.tsa.stattools import adfuller

import statsmodels.graphics.tsaplots as pl

from statsmodels.tsa.statespace.sarimax import SARIMAX 
print(data_covid.info())
delhi_data=data_covid.loc[data_covid['State/UnionTerritory']=='Delhi']

print(delhi_data.info())
df=delhi_data[['Date','Confirmed']]

df['Date']=pd.to_datetime(df['Date'], dayfirst=True)

df.set_index('Date',inplace=True)

df.columns=['Confirmed']

df=df[df['Confirmed']!=0]

print(df.describe())
ax=df.plot(figsize=(10,10),color='Red',fontsize=15)

ax.set_xlabel('Month', fontsize=15)

ax.set_ylabel('Number of Confirmed Cases',fontsize=15)

ax.set_title('Spread of Covid-19: Delhi',fontsize=20)

plt.style.use('fivethirtyeight')

plt.show()
#Checking for Stationarity

res=adfuller(df)

print('p-value=',res[1])
#Convert to stationary series by- 



#Taking log transform  

df_log=np.log(df)



#Plotting stationary series

ax1=df_log.plot()

ax1.set_xlabel('Month', fontsize=15)

ax1.set_ylabel('Number of Confirmed Cases',fontsize=15)

ax1.set_title('Spread of Covid-19: Delhi- Stationary',fontsize=20)

plt.style.use('fivethirtyeight')

plt.show()



#Checking for stationarity 

check=adfuller(df_log)

print('p-value=',check[1])
df_train=df_log.iloc[:len(df)-50]

df_test=df_log.iloc[len(df)-50:]
#Plotting ACF

pl.plot_acf(df_train,lags=10,zero=False)

plt.title('Autocorrelation Plot')

plt.show()

#Plotting PACF

pl.plot_pacf(df_train,lags=10,zero=False)

plt.title('Partial Autocorrelation Plot')

plt.show()
#Finding the best model 

param=[]

for p in range(4):

    for q in range(4):

        model=SARIMAX(df_train,order=(p,0,q),trend='c')

        results=model.fit()

        param.append([p,q,results.aic,results.bic])

res=pd.DataFrame(param,columns=['p','q','aic','bic'])

print(res.sort_values('aic'))

#Training the best model

model=SARIMAX(df_train,order=(2,0,1),trend='c')

results=model.fit()

print(results.aic)

#Mean absolute error

residuals=results.resid

mae=np.mean(np.abs(residuals))

print(mae)
#Plotting diagnostics

results.plot_diagnostics(figsize=(20,20))

plt.show()
#Summary of results

print(results.summary())
#Predicted values

predictions=results.predict(1,df.shape[0])
#Plotting data

plt.figure(figsize=(15,10))

plt.plot(df_log[:len(df)-50],label = "Training values",color='blue')

plt.plot(predictions,color='red',label='Predicted values')

plt.title('ARIMA(2,1) model')

plt.legend()

plt.show()