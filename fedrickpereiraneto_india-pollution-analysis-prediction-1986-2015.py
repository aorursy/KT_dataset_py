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
dataset = pd.read_csv('/kaggle/input/india_pollution.csv',encoding='cp1252')
dataset.head()
dataset = dataset.drop(['stn_code','sampling_date','agency','location','location_monitoring_station'], axis = 1)
dataset.isnull().sum()
dataset.describe()
dataset['date'].describe()
common_value_date='2015-03-19'

dataset['date']=dataset['date'].fillna(common_value_date)

dataset.tail()
dataset['type'].describe()
type_value = { 'type' : 'Residential, Rural and other Areas'}

dataset = dataset.fillna(value = type_value)
dataset[['spm','pm2_5']] = dataset[['spm','pm2_5']].fillna(0)
import numpy as np

from sklearn.impute import SimpleImputer

imp = SimpleImputer(strategy='mean')

imp = imp.fit(dataset.iloc[:,2:5].values)

dataset.iloc[:,2:5] = imp.transform(dataset.iloc[:,2:5].values)
dataset.isnull().sum()
import datetime



dataset['date'] = pd.to_datetime(dataset['date'])
yno2 = dataset.groupby(dataset['date'].dt.strftime('%Y'))['no2'].mean()

yso2 = dataset.groupby(dataset['date'].dt.strftime('%Y'))['so2'].mean()

yrspm = dataset.groupby(dataset['date'].dt.strftime('%Y'))['rspm'].mean()

yspm = dataset.groupby(dataset['date'].dt.strftime('%Y'))['spm'].mean()

ypm2_5 = dataset.groupby(dataset['date'].dt.strftime('%Y'))['pm2_5'].mean()
yno2 = yno2.to_frame()

yso2 = yso2.to_frame()

yrspm = yrspm.to_frame()

yspm = yspm.to_frame()

ypm2_5 = ypm2_5.to_frame()

df = pd.concat([yno2,yso2,yrspm,yspm,ypm2_5],axis=1)

df
df.plot.area()
top_NO2_levels_state = dataset.groupby(['state'])['no2'].mean().sort_values(ascending = False)

top_NO2_levels_state.head()
top_SO2_levels_state = dataset.groupby(['state'])['so2'].mean().sort_values(ascending = False)

top_SO2_levels_state.head()
top_RSPM_levels_state = dataset.groupby(['state'])['rspm'].mean().sort_values(ascending = False)

top_RSPM_levels_state.head()
top_SPM_levels_state = dataset.groupby(['state'])['spm'].mean().sort_values(ascending = False)

top_SPM_levels_state.head()
top_PM2_5_levels_state = dataset.groupby(['state'])['pm2_5'].mean().sort_values(ascending = False)

top_PM2_5_levels_state.head()
import matplotlib.pyplot as plt

import seaborn as sns



fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(20,16))

ax = sns.barplot("so2", y="type",

                 data=dataset,

                 ax=axes[0,0]

                )

ax = sns.barplot("no2", y="type",

                 data=dataset,

                 ax=axes[0,1]

                )

ax = sns.barplot("rspm", y="type",

                 data=dataset,

                 ax=axes[1,0]

                )

ax = sns.barplot("spm", y="type",

                 data=dataset,

                 ax=axes[1,1]

                )
def calculate_si(so2):

    si=0

    if (so2<=40):

     si= so2*(50/40)

    if (so2>40 and so2<=80):

     si= 50+(so2-40)*(50/40)

    if (so2>80 and so2<=380):

     si= 100+(so2-80)*(100/300)

    if (so2>380 and so2<=800):

     si= 200+(so2-380)*(100/800)

    if (so2>800 and so2<=1600):

     si= 300+(so2-800)*(100/800)

    if (so2>1600):

     si= 400+(so2-1600)*(100/800)

    return si

dataset['si']=dataset['so2'].apply(calculate_si)

df= dataset[['so2','si']]

df.head()
#Function to calculate no2 individual pollutant index(ni)

def calculate_ni(no2):

    ni=0

    if(no2<=40):

     ni= no2*50/40

    elif(no2>40 and no2<=80):

     ni= 50+(no2-14)*(50/40)

    elif(no2>80 and no2<=180):

     ni= 100+(no2-80)*(100/100)

    elif(no2>180 and no2<=280):

     ni= 200+(no2-180)*(100/100)

    elif(no2>280 and no2<=400):

     ni= 300+(no2-280)*(100/120)

    else:

     ni= 400+(no2-400)*(100/120)

    return ni

dataset['ni']=dataset['no2'].apply(calculate_ni)

df= dataset[['no2','ni']]

df.head()


def calculate_(rspm):

    rpi=0

    if(rpi<=30):

     rpi=rpi*50/30

    elif(rpi>30 and rpi<=60):

     rpi=50+(rpi-30)*50/30

    elif(rpi>60 and rpi<=90):

     rpi=100+(rpi-60)*100/30

    elif(rpi>90 and rpi<=120):

     rpi=200+(rpi-90)*100/30

    elif(rpi>120 and rpi<=250):

     rpi=300+(rpi-120)*(100/130)

    else:

     rpi=400+(rpi-250)*(100/130)

    return rpi

dataset['rpi']=dataset['rspm'].apply(calculate_si)

df= dataset[['rspm','rpi']]

df.tail()

#many data values of rspm values is unawailable since it was not measure before
def calculate_spi(spm):

    spi=0

    if(spm<=50):

     spi=spm

    if(spm<50 and spm<=100):

     spi=spm

    elif(spm>100 and spm<=250):

     spi= 100+(spm-100)*(100/150)

    elif(spm>250 and spm<=350):

     spi=200+(spm-250)

    elif(spm>350 and spm<=450):

     spi=300+(spm-350)*(100/80)

    else:

     spi=400+(spm-430)*(100/80)

    return spi

dataset['spi']=dataset['spm'].apply(calculate_spi)

df= dataset[['spm','spi']]

df.tail()
#its is calculated as per indian govt standards

def calculate_aqi(si,ni,spi,rpi):

    aqi=0

    if(si>ni and si>spi and si>rpi):

     aqi=si

    if(spi>si and spi>ni and spi>rpi):

     aqi=spi

    if(ni>si and ni>spi and ni>rpi):

     aqi=ni

    if(rpi>si and rpi>ni and rpi>spi):

     aqi=rpi

    return aqi

dataset['AQI']=dataset.apply(lambda x:calculate_aqi(x['si'],x['ni'],x['spi'],x['rpi']),axis=1)

df= dataset[['date','state','si','ni','rpi','spi','AQI']]

df.head()
yearly_AQI_levels = df.groupby(df['date'].dt.strftime('%Y'))['AQI'].mean()

plt.figure(figsize=(16,8))

plt.title('Yearly AQI Levels')

plt.xlabel('Year')

plt.ylabel('AQI Level')

plt.plot(yearly_AQI_levels,marker = 'o');
plt.figure(figsize=(16,6))

plt.plot(yearly_AQI_levels.rolling(window=12,center=False).mean(),label='Rolling Mean');

plt.plot(yearly_AQI_levels.rolling(window=12,center=False).std(),label='Rolling sd');

plt.legend();
import statsmodels.api as sm



# multiplicative

res = sm.tsa.seasonal_decompose(yearly_AQI_levels.values,period=12,model="multiplicative")

plt.figure(figsize=(16,12))

fig = res.plot()
# additive

res = sm.tsa.seasonal_decompose(yearly_AQI_levels.values,period=12,model="additive")

plt.figure(figsize=(16,12))

fig = res.plot()
# Stationarity tests

from statsmodels.tsa.stattools import adfuller



def test_stationarity(timeseries):

    

    #Perform Dickey-Fuller test:

    print('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print (dfoutput)



test_stationarity(yearly_AQI_levels)
df1 = df.groupby(['date'])['AQI'].mean()

df1.isnull().sum()



monthly_AQI_levels = df1.resample("M").mean()

monthly_AQI_levels.head()
monthly_AQI_levels = monthly_AQI_levels.fillna(monthly_AQI_levels.mean())

plt.figure(figsize=(16,8))

plt.title('monthly AQI Levels')

plt.xlabel('Month')

plt.ylabel('AQI Level')

plt.plot(monthly_AQI_levels);
# to remove trend

from pandas import Series as Series

# create a differenced series

def difference(dataset, interval=1):

    diff = list()

    for i in range(interval, len(dataset)):

        value = dataset[i] - dataset[i - interval]

        diff.append(value)

    return Series(diff)



# invert differenced forecast

def inverse_difference(last_ob, value):

    return value + last_ob
plt.figure(figsize=(16,16))

plt.subplot(311)

plt.title('Original')

plt.xlabel('Time')

plt.ylabel('Sales')

plt.plot(monthly_AQI_levels)

plt.subplot(312)

plt.title('After De-trend')

plt.xlabel('Time')

plt.ylabel('Sales')

new_ts=difference(monthly_AQI_levels)

plt.plot(new_ts)

plt.plot()



plt.subplot(313)

plt.title('After De-seasonalization')

plt.xlabel('Time')

plt.ylabel('Sales')

new_ts=difference(monthly_AQI_levels,12)       # assuming the seasonality is 12 months long

plt.plot(new_ts)

plt.plot()
test_stationarity(new_ts)
ts=df1.resample("M").mean()

ts.index=pd.date_range(start = '1987-01-01',end='2015-12-01', freq = 'MS')

ts=ts.reset_index()

ts.head()
from fbprophet import Prophet

#prophet reqiures a pandas df at the below config 

# ( date column named as DS and the value column as Y)

ts.columns=['ds','y']

model = Prophet( yearly_seasonality=True) #instantiate Prophet with only yearly seasonality as our data is monthly 

model.fit(ts) #fit the model with your dataframe
# predict for 3 years in the furure and YS - Year start is the frequency

future = model.make_future_dataframe(periods = 3, freq = 'YS')  

# now lets make the forecasts

forecast = model.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
model.plot(forecast)
model.plot_components(forecast)
forecast.tail()