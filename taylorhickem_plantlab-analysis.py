#import new packages

!pip install psypy



#load the github repository into workspace

!cp -r ../input/plantlab/* ./



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime as dt

import psypy.psySI as si

import os, sys



import data_io
#import the sensor readings table

df = pd.read_sql_table('sensor_readings',data_io.SQL_DBNAME)

df.set_index('datetime',inplace=True)

df = df.loc['2019'].copy()

df[df.sensorid=='TA01'].plot()
window = df.loc['2019-07-18':'2019-07-20'].copy()

window[window.sensorid=='TA01'].plot()
##convert to nearest 10 minute dataset

int_min = 10

window.reset_index(inplace=True)

window['dt_short'] = window.apply(lambda x:dt.datetime(

    year=x.datetime.year,month=x.datetime.month,day=x.datetime.day,

    hour=x.datetime.hour,minute=int(x.datetime.minute/int_min)*int_min),axis=1)

del window['datetime']

window.rename(columns={'dt_short':'datetime'},inplace=True)
#create pivot table

# datetime, value 

tsdata = pd.pivot_table(window,values='value',columns='sensorid',index='datetime',aggfunc='mean')

tsdata.head()
tsdata['ENTH01'] = tsdata.apply(lambda x: si.state("DBT", x.TA01+273.15, "RH", x.HA01/100, 101325)[1],axis=1)

tsdata['MOIST01'] = tsdata.apply(lambda x: si.state("DBT", x.TA01+273.15, "RH", x.HA01/100, 101325)[4],axis=1)

tsdata[['TA01','HA01','ENTH01','MOIST01']].head()
tsdata['ENTH01'].plot()
tsdata['MOIST01'].plot()
tsdata.loc['2019-07-19 08':'2019-07-19 20']['ENTH01'].plot()
tsdata.loc['2019-07-19 08':'2019-07-19 20']['MOIST01'].plot()
tsdata.loc['2019-07-19 08':'2019-07-19 20']['TA01'].plot()
tsdata.loc['2019-07-19 08':'2019-07-19 20']['HA01'].plot()