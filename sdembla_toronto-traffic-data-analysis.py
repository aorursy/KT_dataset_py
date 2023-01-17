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
import numpy as np

import pandas as pd

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

from pandas.plotting import autocorrelation_plot, scatter_matrix



#visualization 

import matplotlib.pyplot as plt

import seaborn as sea



from pandas import DataFrame, Series

import statsmodels.formula.api as sm

from sklearn.linear_model import LinearRegression

import scipy, scipy.stats



import seaborn as sns

%matplotlib inline
train_data = pd.read_csv("/kaggle/input/KSI_CLEAN.csv")

df = pd.DataFrame(train_data)
year = train_data['YEAR']

month = train_data['MONTH']

data = pd.read_csv("/kaggle/input/KSI_CLEAN.csv", parse_dates=[['YEAR', 'MONTH']])

data['YEAR'] = year

data['MONTH'] = month
data['TIMESTAMP'] = pd.to_datetime(data.YEAR_MONTH) + pd.to_timedelta(data.HOUR, unit='h') + pd.to_timedelta(data.MINUTES, unit='m')
df1 = data.replace(' ', np.nan, regex=False)
missing_percent = df1.isna().sum()/len(df)

# missing_percent * 100
data_clean=df1.dropna(axis=1, thresh=3000, how="any")
pivot=data_clean.pivot_table(index='YEAR',margins=True,margins_name='TOTAL',values=['ALCOHOL', 'PEDESTRIAN', 'CYCLIST', 'TRSN_CITY_VEH', 'MOTORCYCLE', 'TRUCK', 'EMERG_VEH', 'AG_DRIV', 'REDLIGHT', 'DISABILITY', 'FATAL', 'SPEEDING'],aggfunc=np.sum)

pivot
ig, ax = plt.subplots(1,1)

pivot.iloc[11].plot(kind='pie', ax=ax, autopct='%3.1f%%', figsize=(8,8), fontsize=15)

ax.set_ylabel('')

ax.set_xlabel('Total Accidents in Ontario in last 10 years(%age)',fontsize=20)
pivot1= data_clean.pivot_table(index='YEAR', margins=False ,values=['ALCOHOL', 'PEDESTRIAN', 'CYCLIST', 'TRSN_CITY_VEH', 'MOTORCYCLE', 'TRUCK', 'EMERG_VEH', 'AG_DRIV', 'REDLIGHT', 'DISABILITY', 'FATAL', 'SPEEDING'],aggfunc=np.sum)
ig, ax = plt.subplots(1,1)

pivot.iloc[0].plot(kind='pie', ax=ax, autopct='%3.1f%%', figsize=(8,8), fontsize=10)

ax.set_ylabel('')

ax.set_xlabel('Total Accidents in Ontario in 2007 (%age)',fontsize=10)
ig, ax = plt.subplots(1,1)

pivot.iloc[1].plot(kind='pie', ax=ax, autopct='%3.1f%%', figsize=(8,8), fontsize=10)

ax.set_ylabel('')

ax.set_xlabel('Total Accidents in Ontario in 2008 (%age)',fontsize=10)
ig, ax = plt.subplots(1,1)

pivot.iloc[2].plot(kind='pie', ax=ax, autopct='%3.1f%%', figsize=(8,8), fontsize=10)

ax.set_ylabel('')

ax.set_xlabel('Total Accidents in Ontario in 2009 (%age)',fontsize=10)
ig, ax = plt.subplots(1,1)

pivot.iloc[3].plot(kind='pie', ax=ax, autopct='%3.1f%%', figsize=(8,8), fontsize=10)

ax.set_ylabel('')

ax.set_xlabel('Total Accidents in Ontario in 2010 (%age)',fontsize=10)
ig, ax = plt.subplots(1,1)

pivot.iloc[4].plot(kind='pie', ax=ax, autopct='%3.1f%%', figsize=(8,8), fontsize=10)

ax.set_ylabel('')

ax.set_xlabel('Total Accidents in Ontario in 2011 (%age)',fontsize=10)
ig, ax = plt.subplots(1,1)

pivot.iloc[5].plot(kind='pie', ax=ax, autopct='%3.1f%%', figsize=(8,8), fontsize=10)

ax.set_ylabel('')

ax.set_xlabel('Total Accidents in Ontario in 2012 (%age)',fontsize=10)
ig, ax = plt.subplots(1,1)

pivot.iloc[6].plot(kind='pie', ax=ax, autopct='%3.1f%%', figsize=(8,8), fontsize=10)

ax.set_ylabel('')

ax.set_xlabel('Total Accidents in Ontario in 2013 (%age)',fontsize=10)
ig, ax = plt.subplots(1,1)

pivot.iloc[7].plot(kind='pie', ax=ax, autopct='%3.1f%%', figsize=(8,8), fontsize=10)

ax.set_ylabel('')

ax.set_xlabel('Total Accidents in Ontario in 2014 (%age)',fontsize=10)
ig, ax = plt.subplots(1,1)

pivot.iloc[8].plot(kind='pie', ax=ax, autopct='%3.1f%%', figsize=(8,8), fontsize=10)

ax.set_ylabel('')

ax.set_xlabel('Total Accidents in Ontario in 2015 (%age)',fontsize=10)
ig, ax = plt.subplots(1,1)

pivot.iloc[9].plot(kind='pie', ax=ax, autopct='%3.1f%%', figsize=(8,8), fontsize=10)

ax.set_ylabel('')

ax.set_xlabel('Total Accidents in Ontario in 2016 (%age)',fontsize=10)
ig, ax = plt.subplots(1,1)

pivot.iloc[10].plot(kind='pie', ax=ax, autopct='%3.1f%%', figsize=(8,8), fontsize=10)

ax.set_ylabel('')

ax.set_xlabel('Total Accidents in Ontario in 2017 (%age)',fontsize=10)
aggressive1= data_clean.pivot_table(index='YEAR', margins=False ,values=['ALCOHOL', 'AG_DRIV', 'SPEEDING'],aggfunc=np.sum)

aggressive1.plot(figsize=(10,8), title="Accidents caused by aggressive driving vs. speeding vs. driving under the influence", grid=True)

plt.ylabel('Accidents')
vehicle_data=data_clean.pivot_table(index='YEAR',margins=True,margins_name='TOTAL',values=['CYCLIST', 'TRSN_CITY_VEH', 'MOTORCYCLE', 'TRUCK', 'EMERG_VEH'],aggfunc=np.sum)

vehicle_data1=data_clean.pivot_table(index='YEAR',margins=False,values=['CYCLIST', 'TRSN_CITY_VEH', 'MOTORCYCLE', 'TRUCK', 'EMERG_VEH'],aggfunc=np.sum)





vehicle_data1.plot(figsize=(10,8), title="Type of vehicles involved in the accidents per year", grid=True)

plt.ylabel('Accidents')
Cyclist_data1=data_clean.pivot_table(index='YEAR',margins=False,values=['CYCLIST'],aggfunc=np.sum)





Cyclist_data1.plot(figsize=(10,8), title="Number of accidents involving cyclists", grid=True)

plt.ylabel('Accidents')
ig, ax = plt.subplots(1,1)

vehicle_data.iloc[11].plot(kind='pie', ax=ax, autopct='%3.1f%%', figsize=(8,8), fontsize=10)

ax.set_ylabel('')

ax.set_xlabel('Vehicle involved in accidents in last 10 years(%age)',fontsize=20)
bins = [0, 4, 8, 12, 16, 20, np.inf]

labels = ['12AM-4AM', '4AM-8AM','8AM-12PM', '12PM-4PM', '4PM-8PM', '8PM-12PM']

data_clean["TIMEOFDAY"] = pd.cut(data_clean["HOUR"], bins, labels = labels)
time_day = pd.DataFrame()

time_day['12AM-4AM'] = data_clean['YEAR'][data_clean['TIMEOFDAY']=='12AM-4AM'].value_counts()

time_day['4AM-8AM'] = data_clean['YEAR'][data_clean['TIMEOFDAY']=='4AM-8AM'].value_counts()

time_day['8AM-12PM'] = data_clean['YEAR'][data_clean['TIMEOFDAY']=='8AM-12PM'].value_counts()

time_day['12PM-4PM'] = data_clean['YEAR'][data_clean['TIMEOFDAY']=='12PM-4PM'].value_counts()

time_day['4PM-8PM'] = data_clean['YEAR'][data_clean['TIMEOFDAY']=='4PM-8PM'].value_counts()

time_day['8PM-12PM'] = data_clean['YEAR'][data_clean['TIMEOFDAY']=='8PM-12PM'].value_counts()

time_day.loc['Total']= time_day.sum()
time_day
ig, ax = plt.subplots(1,1)

time_day.iloc[11].plot(kind='pie', ax=ax, autopct='%3.1f%%', figsize=(8,8), fontsize=10)

ax.set_ylabel('')

ax.set_xlabel('Time of the day vs total number accidents (%age)',fontsize=20)
time_of_day = data_clean.groupby('TIMEOFDAY').count()

time_of_day['TOTAL'] = time_of_day['YEAR_MONTH']

time_of_day = time_of_day[['TOTAL']]
sea.barplot(x="TIMEOFDAY", y="TOTAL", data=time_of_day.reset_index())

plt.title = 'Intel'

plt.show()
location = pd.DataFrame()

location['Etobicoke'] = data_clean['YEAR'][data_clean['District']=='Etobicoke York'].value_counts()

location['NorthYork'] = data_clean['YEAR'][data_clean['District']=='North York'].value_counts()

location['Scarborough'] = df1['YEAR'][df1['District']=='Scarborough'].value_counts()

location['EastYork'] = df1['YEAR'][df1['District']=='Toronto East York'].value_counts()

location.loc['Total']= location.sum()
location
location1 = pd.DataFrame()

location1['Etobicoke'] = data_clean['YEAR'][data_clean['District']=='Etobicoke York'].value_counts()

location1['NorthYork'] = data_clean['YEAR'][data_clean['District']=='North York'].value_counts()

location1['Scarborough'] = df1['YEAR'][df1['District']=='Scarborough'].value_counts()

location1['EastYork'] = df1['YEAR'][df1['District']=='Toronto East York'].value_counts()

result = location1.sort_index(inplace=True)

location1.plot(figsize=(10,8), title="Number of accidents in regions of GTA", grid=True)

plt.ylabel('Accidents')
ig, ax = plt.subplots(1,1)

location.iloc[11].plot(kind='pie', ax=ax, autopct='%3.1f%%', figsize=(10,10), fontsize=15)

ax.set_ylabel('')

ax.set_xlabel('Total Accidents in Toronto region wise data in last 10 years(%age)',fontsize=20)
ig, ax = plt.subplots(1,1)

location.iloc[10].plot(kind='pie', ax=ax, autopct='%3.1f%%', figsize=(10,10), fontsize=15)

ax.set_ylabel('')

ax.set_xlabel('Total Accidents in Toronto region wise data in 2017(%age)',fontsize=20)
ig, ax = plt.subplots(1,1)

location.iloc[9].plot(kind='pie', ax=ax, autopct='%3.1f%%', figsize=(10,10), fontsize=15)

ax.set_ylabel('')

ax.set_xlabel('Total Accidents in Toronto region wise data in 2016(%age)',fontsize=20)
ig, ax = plt.subplots(1,1)

location.iloc[8].plot(kind='pie', ax=ax, autopct='%3.1f%%', figsize=(10,10), fontsize=15)

ax.set_ylabel('')

ax.set_xlabel('Total Accidents in Toronto region wise data in 2015(%age)',fontsize=20)
ig, ax = plt.subplots(1,1)

location.iloc[7].plot(kind='pie', ax=ax, autopct='%3.1f%%', figsize=(10,10), fontsize=15)

ax.set_ylabel('')

ax.set_xlabel('Total Accidents in Toronto region wise data in 2014(%age)',fontsize=20)
ig, ax = plt.subplots(1,1)

location.iloc[6].plot(kind='pie', ax=ax, autopct='%3.1f%%', figsize=(10,10), fontsize=15)

ax.set_ylabel('')

ax.set_xlabel('Total Accidents in Toronto region wise data in 2013(%age)',fontsize=20)
ig, ax = plt.subplots(1,1)

location.iloc[5].plot(kind='pie', ax=ax, autopct='%3.1f%%', figsize=(10,10), fontsize=15)

ax.set_ylabel('')

ax.set_xlabel('Total Accidents in Toronto region wise data in 2012(%age)',fontsize=20)
pivot_time=data_clean.pivot_table(index='YEAR_MONTH',margins=True,margins_name='TOTAL',values=['ALCOHOL', 'PEDESTRIAN', 'CYCLIST', 'TRSN_CITY_VEH', 'MOTORCYCLE', 'TRUCK', 'EMERG_VEH', 'AG_DRIV', 'REDLIGHT', 'DISABILITY', 'FATAL', 'SPEEDING'],aggfunc=np.sum)

# pivot_time
time_series_ad = pd.DataFrame()

time_series_ad['present'] = pivot_time['AG_DRIV']

time_series_ad['shift'] = pivot_time['AG_DRIV'].shift(1)

time_series_ad['change'] = (pivot_time['AG_DRIV'] - pivot_time['AG_DRIV'].shift(1))*100/pivot_time['AG_DRIV'].shift(1)

# time_series_ad
plt.figure()

autocorrelation_plot(time_series_ad['change'].dropna())

plt.title ='AG_DRIV'
time_series_pd = pd.DataFrame()

time_series_pd['present'] = pivot_time['PEDESTRIAN']

time_series_pd['shift'] = pivot_time['PEDESTRIAN'].shift(1)

time_series_pd['change'] = (pivot_time['PEDESTRIAN'] - pivot_time['PEDESTRIAN'].shift(1))*100/pivot_time['PEDESTRIAN'].shift(1)

# time_series_pd
plt.figure()

autocorrelation_plot(time_series_pd['change'].dropna())

plt.title ='PEDESTRIAN'
sns.pairplot(pivot1)