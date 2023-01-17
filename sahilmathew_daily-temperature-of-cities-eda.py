import numpy as np

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt



%matplotlib inline
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
temp=pd.read_csv("/kaggle/input/daily-temperature-of-major-cities/city_temperature.csv")
temp.head()
temp['Region'].unique()
temp['Country'].unique()
temp['Month'].unique()
temp['Day'].unique()
temp['Year'].unique()
temp['AvgTemperature']
sns.distplot(temp['AvgTemperature'],kde=False)

#there's some extremely low temp, I'll just get rid of it. 
temp.head(3)
temp[temp['AvgTemperature'] < -80]

temp.drop(temp[temp['AvgTemperature']<-80].index,inplace=True)
sns.distplot(temp['AvgTemperature'],bins=100,kde=False)

#removed the outliers 
#city with the highest temperature 

temp[temp['AvgTemperature']==temp['AvgTemperature'].max()]
#city with the lowest temperature

temp[temp['AvgTemperature']==temp['AvgTemperature'].min()]
#graph of temperatures in different regions 

#grouping all the regions together 

regions=temp.groupby('Region')

regions.describe()
africa=temp[temp['Region']=='Africa']
sns.distplot(africa['AvgTemperature'],kde=False)
asia=temp[temp['Region']=='Asia']
sns.distplot(asia['AvgTemperature'],kde=False)
australia=temp[temp['Region']=='Australia/South Pacific']
sns.distplot(australia['AvgTemperature'],kde=False)
europe=temp[temp['Region']=='Europe']
sns.distplot(europe['AvgTemperature'],kde=False)
middleeast=temp[temp['Region']=='Middle East']
sns.distplot(middleeast['AvgTemperature'],kde=False,bins=100)
america=temp[temp['Region']=='North America']
sns.distplot(america['AvgTemperature'],kde=False,bins=100)
south_america=temp[temp['Region']=='South/Central America & Carribean']
sns.distplot(south_america['AvgTemperature'],kde=False,bins=100)
temp.head()
#fig=plt.figure()



#axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)



#we need to aggregate a date coloumn temp.head()
temp.drop('Date',axis=1,inplace=True)
temp.head()
import datetime
temp['Date']=pd.to_datetime((temp.Year*10000+temp.Month*100+temp.Day).apply(str),format='%Y%m%d')
temp.drop(['Month','Day','Year'],axis=1,inplace=True)

#The time coloumns have been converted to a single Date col
temp.head()
africa=temp[temp['Region']=='Africa']
africa['Year']=africa['Date'].map(lambda x:x.year)
africa.head()
af=africa.groupby('Year').mean().reset_index().drop(25)

af.head()
sns.lineplot(x='Year',y='AvgTemperature',data=af)
asia=temp[temp['Region']=='Asia']

asia['Year']=asia['Date'].map(lambda x:x.year)

asia1=asia.groupby('Year').mean().reset_index().drop(25)

asia1.head()
sns.lineplot(x='Year',y='AvgTemperature',data=asia1)
australia=temp[temp['Region']=='Australia/South Pacific']

australia['Year']=australia['Date'].map(lambda x:x.year)

aus=australia.groupby('Year').mean().reset_index().drop(25)
sns.lineplot(x='Year',y='AvgTemperature',data=aus)
europe=temp[temp['Region']=='Europe']

europe['Year']=europe['Date'].map(lambda x:x.year)

eu=europe.groupby('Year').mean().reset_index().drop(25)

sns.lineplot(x='Year',y='AvgTemperature',data=eu)
middleeast=temp[temp['Region']=='Middle East']

middleeast['Year']=middleeast['Date'].map(lambda x:x.year)

mid=middleeast.groupby('Year').mean().reset_index().drop(25)

sns.lineplot(x='Year',y='AvgTemperature',data=mid)
south_america=temp[temp['Region']=='South/Central America & Carribean']

south_america['Year']=south_america['Date'].map(lambda x:x.year)

sa=south_america.groupby('Year').mean().reset_index().drop(25)

sns.lineplot(x='Year',y='AvgTemperature',data=sa)
america=temp[temp['Region']=='North America']

america['Year']=america['Date'].map(lambda x:x.year)

am=america.groupby('Year').mean().reset_index().drop(25)

sns.lineplot(x='Year',y='AvgTemperature',data=am)