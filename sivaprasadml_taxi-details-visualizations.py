# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
source = pd.read_csv("../input/2014_15_txdet3.csv")



df = pd.DataFrame()

df = source

df.shape
unwanted_indices = df[(df['pickup_country'] != 'US') | (df['dropoff_country'] != 'US')].index

df.drop(list(unwanted_indices), inplace=True)

df.shape
unwanted_indices = df[ (df['fare_amount'] < 0) | 

                       (df['fare_amount'] == 0) ].index

df.drop(list(unwanted_indices), inplace=True)

df.reset_index()

df.shape

unwanted_indices = df[ (df['distance'] == 0)].index

df.drop(list(unwanted_indices), inplace=True)

df.reset_index(drop=True)

df.shape

df.drop(['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','pickup_country','dropoff_country'],axis=1,inplace=True)

df.boxplot('fare_amount',figsize=(20,8))

q1 = df.fare_amount.quantile(0.25)

q3 = df.fare_amount.quantile(0.75)

iqr = q3 - q1

l_lim = q1 - (1.5 * iqr)

u_lim = q3 + (1.5 * iqr)

print('q1:', q1, 'q3:', q3,'iqr:', iqr,'l_lim:',l_lim,'u_lim',u_lim)
unwanted_indices = df[ (df['fare_amount'] > u_lim) ].index

df.drop(list(unwanted_indices), inplace=True)

df.reset_index()

df.shape
df.boxplot('fare_amount',figsize=(20,8))
df.boxplot('distance',figsize=(20,8))

q1 = df.distance.quantile(0.25)

q3 = df.distance.quantile(0.75)

iqr = q3 - q1

l_lim = q1 - (1.5 * iqr)

u_lim = q3 + (1.5 * iqr)

print('q1:', q1, 'q3:', q3,'iqr:', iqr,'l_lim:',l_lim,'u_lim',u_lim)
unwanted_indices = df[ (df['distance'] > u_lim) ].index

df.drop(list(unwanted_indices), inplace=True)
df.boxplot('distance', figsize=(20,8))
df.corr()
df.dtypes
'''max_date = pd.to_datetime(df['key'].max())

min_date = pd.to_datetime(df['key'].min())





from pandas.tseries.holiday import USFederalHolidayCalendar as calendar



dr = pd.date_range(start=min_date, end=max_date)



cal = calendar()

holidays = cal.holidays(start=dr.min(), end=dr.max())

#df1['holiday'] = df1['date'].isin(holidays)



df['holiday'] = pd.to_datetime(df['key']).isin(holidays)'''
df['hour'] = df.pickup_datetime.apply(lambda x: x[11:13])

df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])





df['month'] = df['pickup_datetime'].dt.month
df.to_csv('final_taxi_det',index=False)
tbl_fare_avg = pd.pivot_table(df, values=[ 'fare_amount','distance'] , index=['year'], aggfunc=np.mean)

tbl_fare_avg
avgfr = tbl_fare_avg.plot.bar(figsize=(20,8))

avgfr.set_xlabel('Year')

avgfr.set_ylabel('Scale')


df_2014 = df[df['year'] == 2014]

df_2014.groupby(['month'])['fare_amount','distance'].mean()
avgMn = df_2014.groupby(['month'])['fare_amount','distance'].mean().plot.bar(figsize=(20,8))

avgMn.set_xlabel('Year')

avgMn.set_ylabel('Scale')

df_2014_09 = df[(df['year'] == 2014) & (df['month'] == 9)]

avgday = df_2014_09.groupby(['date'])['fare_amount','distance'].mean().plot.bar(figsize=(20,8))

avgday.set_xlabel('day')

avgday.set_ylabel('Scale')

df_2014_day = df[(df['date'] == '2014-09-29') ]
table_hr_sum = pd.pivot_table(df_2014_day, values=[ 'fare_amount','distance' ] , index=['hour'], aggfunc=np.mean)

table_hr_sum



df_hr_sum = pd.DataFrame(table_hr_sum)





plt.figure(figsize=(15,7))



plt.plot(df_hr_sum.index, df_hr_sum['distance'], color = 'green')

plt.plot(df_hr_sum.index, df_hr_sum['fare_amount'], color = 'orange')

#plt.plot(df_hr_sum.index, df_hr_sum['passenger_count'], color = 'red')



plt.xlabel('Hours')

plt.ylabel('')

plt.title('2014-09-29 - trip details')

plt.legend()

plt.show()
df_2014_day = df[(df['date'] == '2014-12-25') ]

table_hr_sum = pd.pivot_table(df_2014_day, values=[ 'fare_amount','distance' ] , index=['hour'], aggfunc=np.mean)



table_hr_sum



df_hr_sum = pd.DataFrame(table_hr_sum)

plt.figure(figsize=(15,7))



plt.plot(df_hr_sum.index, df_hr_sum['distance'], color = 'green')

plt.plot(df_hr_sum.index, df_hr_sum['fare_amount'], color = 'orange')

#plt.plot(df_hr_sum.index, df_hr_sum['passenger_count'], color = 'red')



plt.xlabel('Hours')

plt.ylabel('')

plt.title('2014-12-25 - trip details')

plt.legend()

plt.show()