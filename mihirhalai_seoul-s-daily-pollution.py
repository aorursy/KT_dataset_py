

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from datetime import datetime as dt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

def dateTimeParser(x):

    return dt.strptime(x,'%Y-%m-%d %H:00')



pollution = pd.read_csv('/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Measurement_summary.csv',

                       date_parser = dateTimeParser, parse_dates=[0],index_col = 0)

pollution.drop('Address',axis=1,inplace=True)

pollution.head()
tot = pollution.iloc[:,3:].sum(axis=1)

total = pd.concat([pollution['Station code'],tot],axis=1)

total.columns.values[-1] = 'total'

stationMeans = total.groupby(['Station code']).agg('mean').sort_values(by='total',ascending=False)

stationMeans.columns = ['Mean']

stationMeans.plot.bar(figsize=(10,5));
def plot_pollutant_daily(pol):

    toxin = pollution[pol].groupby(pollution.index.date).agg('mean')

    mean = toxin.rolling(10).mean()

    

    plt.figure(figsize=(20,10))

    plt.title('Daily level of {} with 10 day rolling average'.format(pol),fontsize=20)

    plt.plot(toxin,linewidth = 0.3,color='blue')

    plt.plot(mean,color='purple',linewidth=2)

    plt.xlabel('date')

    plt.ylabel('level')


plot_pollutant_daily('PM2.5')

plot_pollutant_daily('PM10')

plot_pollutant_daily('CO')

plot_pollutant_daily('O3')

plot_pollutant_daily('NO2')

plot_pollutant_daily('SO2')
#daily percentage increase/decrease

allpolls = pollution.iloc[:,3:].groupby(pollution.index.date).agg('mean').pct_change(axis=0)

plt.figure(figsize = (20,10))



cols = allpolls.columns.values

for c in cols:

    plt.plot(allpolls[c],label = c)

plt.title('Daily percentage increase',fontsize=20)

plt.xlabel('date',fontsize=14)

plt.ylabel('%',fontsize=14)

plt.legend();