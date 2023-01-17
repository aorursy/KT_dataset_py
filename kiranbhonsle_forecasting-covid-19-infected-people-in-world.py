import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.metrics import mean_absolute_error,mean_squared_error

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.ar_model import AR

from datetime import datetime

from statsmodels.tsa.holtwinters import ExponentialSmoothing

import math

from statsmodels.tsa.statespace.sarimax import SARIMAX

from keras.preprocessing.sequence import TimeseriesGenerator

from statsmodels.tsa.seasonal import seasonal_decompose
cor_inf = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

cor_inf.head(5)
#drop lat long and province/state columns

cor_inf.columns
cor_inf.drop(labels = ['Province/State','Lat', 'Long'],axis = 1, inplace= True)
cor_inf.head(20)
cor_inf.shape
#cor_inf['Country/Region'].value_counts()
#group by data based upon country since the countries name are repeated more than once 

cor_inf = cor_inf.groupby(['Country/Region']).sum()
cor_inf.loc['China'].tail(5)
#reshape the data as per the time series analysis

cor_inf_re = pd.DataFrame()

for i in range(0,len(cor_inf)):

    cor_inf_re[cor_inf.index[i]] = cor_inf.iloc[i].values

    
type(cor_inf.index[0])
cor_inf_re.index = cor_inf.columns[:]
cor_inf_re.head(5)
def total_infected_sum():

    count = []

    for i in range(0,len(cor_inf_re)):

        count.append(sum(cor_inf_re.iloc[i].values))

    return count
cor_inf_re['Total infected'] = total_infected_sum()
cor_inf_re.tail(5)
def parser(date):

    date = datetime.strptime(date,'%m/%d/%y')

    date  = str(date.day) + '-' + str(date.month) + '-' + str(date.year)

    print(date)

    return datetime.strptime(date,'%d-%m-%Y')
#convert str to datetime in index 

timestamp = []

for i in range(0,len(cor_inf_re)):

    timestamp.append(parser(cor_inf_re.index[i]))

cor_inf_re.index = timestamp
cor_inf_re.to_csv('./covid_19_confirmed.csv')
#preparing for time series

infected_people = cor_inf_re['Total infected']
#column for infected per day

diff = []

diff.append(cor_inf_re['Total infected'][0])

for i in range(0,len(cor_inf_re['Total infected']) - 1):

    diff.append(cor_inf_re['Total infected'][i+1] - cor_inf_re['Total infected'][i])



cor_inf_re['Infected_per_Day'] = diff
#visualization

plt.xlabel('dates')

plt.ylabel('infected people')

infected_people.plot(figsize = (11,5),marker='o')

plt.legend()
#check the statistical part of the data

infected_people.describe()
#to check if there is an trend or seasonality

from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(infected_people)
result.trend.plot(figsize=(12,4))
result.seasonal.plot(figsize=(12,4))
#autocorrelation graph

plot_acf(infected_people)
plot_pacf(infected_people)
infec_one = infected_people.diff(periods=1)

infec_one = infec_one[1:]

plot_acf(infec_one)
train = infected_people.iloc[:-8]

test = infected_people.iloc[-8:]
model = ExponentialSmoothing(train,trend = "mul",seasonal_periods=7,seasonal="add").fit()
predictions = model.predict(start = 50 ,end= 57 )

#predictions
plt.figure(figsize = (12,4))

predictions.plot(c ='r',marker = 'o',markersize=10,linestyle='--')

test.plot(marker = 'o',markersize=10,linestyle='--')

print("root mean squared error : ",math.sqrt(mean_squared_error(test,predictions)))

print("mean absolute error : ",mean_absolute_error(test,predictions))
model = SARIMAX(train,order = (4,2,1),trend='t',seasonal_order=(2, 2, 1, 14))

model_fit = model.fit()
predictions = model_fit.predict(start = 53,end=60)

predictions
plt.figure(figsize = (12,4))

plt.plot(predictions,'r',marker = 'o',markersize=10,linestyle='--')

plt.plot(test,marker = 'o',markersize=10,linestyle='--')

print("root mean squared error : ",math.sqrt(mean_squared_error(test,predictions)))

print("mean absolute error : ",mean_absolute_error(test,predictions))
predictions = model_fit.predict(start = 50,end=67)

predictions