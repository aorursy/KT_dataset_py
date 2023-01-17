import numpy as np

import pylab as pl

import pandas as pd

import matplotlib.pyplot as plt 

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/train.csv")

test  = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/test.csv")
train.head()
test.head()
fig = plt.figure(figsize=(16,8))

ax = fig.add_subplot(111)

train.groupby('Date').mean().sort_values(by='ConfirmedCases', ascending=False)['ConfirmedCases'].plot('bar', color='r',width=0.3,title='Date Confirmed Cases', fontsize=10)

plt.xticks(rotation = 90)

plt.ylabel('Date')

ax.title.set_fontsize(30)

ax.xaxis.label.set_fontsize(10)

ax.yaxis.label.set_fontsize(10)

print(train.groupby('Date').mean().sort_values(by='ConfirmedCases', ascending=False)['ConfirmedCases'][[1,2]])

print(train.groupby('Date').mean().sort_values(by='ConfirmedCases', ascending=False)['ConfirmedCases'][[4,5,6]])
fig = plt.figure(figsize=(16,8))

ax = fig.add_subplot(111)

train.groupby('Country_Region').mean().sort_values(by='Fatalities', ascending=False)['Fatalities'].plot('bar', color='r',width=0.3,title='Country Region Fatalities', fontsize=9)

plt.xticks(rotation = 90)

plt.ylabel('Confirmed Cases')

ax.title.set_fontsize(30)

ax.xaxis.label.set_fontsize(20)

ax.yaxis.label.set_fontsize(10)

print(train.groupby('Country_Region').mean().sort_values(by='Fatalities', ascending=False)['Fatalities'][[1,2]])

print(train.groupby('Country_Region').mean().sort_values(by='Fatalities', ascending=False)['Fatalities'][[4,5,6]])
#US

ConfirmedCases_date_US = train[train['Country_Region']=='US'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_date_US = train[train['Country_Region']=='US'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_US = ConfirmedCases_date_US.join(fatalities_date_US)





#China

ConfirmedCases_date_China = train[train['Country_Region']=='China'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_date_China = train[train['Country_Region']=='China'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_China = ConfirmedCases_date_China.join(fatalities_date_China)



#Italy

ConfirmedCases_date_Italy = train[train['Country_Region']=='Italy'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_date_Italy = train[train['Country_Region']=='Italy'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_Italy = ConfirmedCases_date_Italy.join(fatalities_date_Italy)



#Australia

ConfirmedCases_date_Australia = train[train['Country_Region']=='Australia'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_date_Australia = train[train['Country_Region']=='Australia'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_Australia = ConfirmedCases_date_Australia.join(fatalities_date_Australia)







plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total_date_US.plot(ax=plt.gca(), title='US')

plt.ylabel("Confirmed  cases", size=13)



plt.subplot(2, 2, 2)

total_date_China.plot(ax=plt.gca(), title='China')



plt.subplot(2, 2, 3)

total_date_Italy.plot(ax=plt.gca(), title='Italy')

plt.ylabel("Confirmed cases", size=13)



plt.subplot(2, 2, 4)

total_date_Australia.plot(ax=plt.gca(), title='Australia')
#Indonesia

ConfirmedCases_date_Indonesia = train[train['Country_Region']=='Indonesia'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_date_Indonesia = train[train['Country_Region']=='Indonesia'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_Indonesia = ConfirmedCases_date_Indonesia.join(fatalities_date_Indonesia)





#Malaysia

ConfirmedCases_date_Malaysia = train[train['Country_Region']=='Malaysia'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_date_Malaysia = train[train['Country_Region']=='Malaysia'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_Malaysia = ConfirmedCases_date_Malaysia.join(fatalities_date_Malaysia)



#Thailand

ConfirmedCases_date_Thailand = train[train['Country_Region']=='Thailand'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_date_Thailand = train[train['Country_Region']=='Thailand'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_Thailand = ConfirmedCases_date_Thailand.join(fatalities_date_Thailand)



#Singapore

ConfirmedCases_date_Singapore = train[train['Country_Region']=='Singapore'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_date_Singapore = train[train['Country_Region']=='Singapore'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_Singapore = ConfirmedCases_date_Singapore.join(fatalities_date_Singapore)







plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total_date_Indonesia.plot(ax=plt.gca(), title='Indonesia')

plt.ylabel("Confirmed  cases", size=13)



plt.subplot(2, 2, 2)

total_date_Malaysia.plot(ax=plt.gca(), title='Malaysia')



plt.subplot(2, 2, 3)

total_date_Thailand.plot(ax=plt.gca(), title='Thailand')

plt.ylabel("Confirmed cases", size=13)



plt.subplot(2, 2, 4)

total_date_Singapore.plot(ax=plt.gca(), title='Singapore')
sub= pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/submission.csv")

sub.to_csv('submission.csv', index = False )