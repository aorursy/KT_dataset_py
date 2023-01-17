# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

dtype = {'DayOfWeek' : np.uint8,
        'DayofMonth' : np.uint8,
        'Month' : np.uint8,
        'Cancelled' : np.uint8,
        'Year' : np.uint16,
        'FlightNum' : np.uint16,
        'Distance' : np.uint16,
        'UniqueCarrier' : str,
        'CancellationCode' : str,
        'Origin' : str,
        'Dest' : str,
        'ArrDelay' : np.float16,
        'DepDelay' : np.float16,
        'CarrierDelay' : np.float16,
        'WeatherDelay' : np.float16,
        'NASDelay' : np.float16,
        'SecurityDelay' : np.float16,
        'LateAircraftDelay' : np.float16,
        'DepTime' : np.float16}
#Not: Dikkat edersen data type a göre sıralılar. np.uint8, np.uint16, str, np.float16 seklinde...
df = pd.read_csv('../input/2008.csv', usecols=dtype.keys(), dtype=dtype)
df.head()

#Sirasi onemli degil, su uc kolonu asagidaki gibi verdigin zaman YYYY-MM-DD seklinde bir datetime64 object olusuyor.
df['Date'] = pd.to_datetime(df.rename(columns = {'DayofMonth' : 'Day'})[['Day', 'Month', 'Year']])

# Any results you write to the current directory are saved as output.
print(df.shape)
print(df.columns)
df.info()
df.head().T
df.info()
df.describe().T
df['UniqueCarrier'].nunique()
#df.groupby('UniqueCarrier').size().sort_values(ascending=False).plot(kind='bar')
#df['UniqueCarrier'].value_counts().plot(kind='bar')
sns.countplot(x='UniqueCarrier', data=df)
df.groupby(['UniqueCarrier', 'FlightNum'])['Distance'].sum().sort_values(ascending=False).iloc[:3]
df.groupby(['UniqueCarrier', 'FlightNum']).agg({'Distance' : [np.mean, np.sum, 'count'],
                                               'Cancelled' : [np.sum]}).sort_values(('Distance', 'sum'), ascending=False).iloc[:10]
pd.crosstab(df['Month'], df['DayOfWeek'])
plt.imshow(pd.crosstab(df.Month, df.DayOfWeek), cmap='seismic', interpolation='none')
#seismic'de kirmizilar en çok, lacivertler en azlar oluyor.
df.hist('Distance', bins=20)
num_flights_by_date = df.groupby('Date').size()
num_flights_by_date.plot(figsize=(20,5))
#Window'u 7nin kati yapinca weekly effect ortadan kalkiyor.
#Cunku window=7k olunca her calculationda haftanin her gununden k kadar dahil ediliyor.
num_flights_by_date.rolling(window=20).mean().plot()
df[df['Cancelled'] == 0].groupby('UniqueCarrier').size().sort_values(ascending=False).iloc[:10]
#EV is not on the list.
#Q1Answer:EV
df['CancellationCode'].value_counts()
#Most frequent reason for cancellation is weather.
#Q2Answer:Weather. Check again, number was different on the website of this data.
#NY: JFK, LGA
#DC: DCA, IAD
#BA: BWI (Baltimore)
#SF: SFO
#LA: LAX
#DA: DFW, DAL, RBD
#SJ: SJC

airports = ['JFK', 'LGA', 'DCA', 'IAD', 'BWI', 'SFO', 'LAX', 'DFW', 'DAL', 'RBD', 'SJC']
num_flights = df[df['Origin'].isin(airports) & df['Dest'].isin(airports)].groupby(['Origin', 'Dest']).size().sort_values(ascending=False)

d = {'JFK' : 'New York',
    'LGA' : 'New York',
    'DCA' : 'Washington',
    'IAD' : 'Washington',
    'BWI' : 'Baltimore',
    'SFO' : 'San Francisco',
    'LAX' : 'Los Angeles',
    'DFW' : 'Dallas',
    'DAL' : 'Dallas',
    'RBD' : 'Dallas',
    'SJC' : 'San Jose'}

num_flights_df = num_flights.reset_index(level=0, inplace=False).reset_index(level=0, inplace=False)
num_flights_df.columns = ['Dest', 'Origin', 'NumFlights']
num_flights_by_city = num_flights_df.replace({'Origin' : d,
                    'Dest' : d})

num_flights_by_city.groupby(['Dest', 'Origin'])['NumFlights'].sum().sort_values(ascending=False)
#Q3Answer: New York - Washington is the most frequent.
#consider cancelled ones.
#df[df['DepDelay'] > 0].count() #2,700,974
#df[df['WeatherDelay'] > 0].count() #99,985
#df[(df['DepDelay'] > 0) & (df['WeatherDelay'] > 0)].count() #99,880
#df[df['DepDelay'] > 0].groupby(['Origin', 'Dest']).size().sort_values(ascending=False) #TOP-5 DELAYED ROUTES
#TOP-5 DELAYED ROUTES:
#LAX     SFO     6253
#DAL     HOU     5742
#SFO     LAX     5322
#ORD     LGA     5311
#HOU     DAL     5288

top5df = df[((df['Origin'] == 'LAX') & (df['Dest'] == 'SFO')) |\
  ((df['Origin'] == 'DAL') & (df['Dest'] == 'HOU')) |\
   ((df['Origin'] == 'SFO') & (df['Dest'] == 'LAX')) |\
   ((df['Origin'] == 'ORD') & (df['Dest'] == 'LGA')) |\
   ((df['Origin'] == 'HOU') & (df['Dest'] == 'DAL'))] #57,504 rows
top5df[top5df['WeatherDelay'] > 0].count()
#Q4Answer:668
#DepTime'ın min değeri 1, max değeri 2400.
#Bucketlar (0-100], (100-200], (200-300], ... ,(2300-2400] olacak.
#Labellar da 0, 1, 2, ... ,23 olacak.
df['DepHour'] = pd.cut(x=df['DepTime'], bins=[0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400], \
      labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'])
df.groupby('DepHour').size().plot(kind='bar')
#Q5Answer: ?
_, axes = plt.subplots(1, 2, figsize=(10, 6))
df.groupby('DayOfWeek').size().plot(kind='bar', ax=axes[0])
df.groupby('Month').size().plot(kind='bar', ax=axes[1])
#Q6Answer:Num of weekend flights are less than weekdays. (YES)
#The lowest number of flights is on Sunday. (NO)
#There are less flights during winter than during summer. (YES)
sns.countplot(x='Month', hue='CancellationCode', data=df[df['Cancelled'] == 1])
#A Carrier
#B Weather
#C National Air System
#D Security
#Q7Answer:December has the highest rate of cancellations due to weather. (YES)
#The highest rate of cancellation in september is due to security. (NO)
#April's top cancellation reason is carriers. (YES)
#Flight cancellations due to NAS are more frequent than those of carriers. (NO)
#Q8Answer:Which month has tha highest carrier cancellations? APRIL
#Q9:Carrier with the greatest number of cancellations due to carrier in April? AA
df[(df['Cancelled'] == 1) & (df['Month'] == 4) & (df['CancellationCode'] == 'A')].groupby('UniqueCarrier').size().plot(kind='bar')
#sns.boxplot(x='UniqueCarrier', y='ArrDelay', data=df[df['ArrDelay'] > 0])
#Yukardaki gibi yapınca outlierlar yüzünden doğru dürüst bir box plot göremedim.
#df2 = df with positive arrival delays
#df3 = df with positive departure delays
df1 = df[df['ArrDelay'] > 0] #positive delay times
df2 = df[df['DepDelay'] > 0]
df1 = df1[df1.ArrDelay < df1.ArrDelay.quantile(0.95)] #remove top .05
df2 = df2[df2.ArrDelay < df2.ArrDelay.quantile(0.95)]
order1 = df1.groupby('UniqueCarrier')['ArrDelay'].mean().sort_values(ascending=False).index
order2 = df2.groupby('UniqueCarrier')['DepDelay'].mean().sort_values(ascending=False).index
_, axes = plt.subplots(2, 1, figsize=(12, 4))
sns.boxplot(x='UniqueCarrier', y='ArrDelay', data=df1, order=order1, ax=axes[0])
sns.boxplot(x='UniqueCarrier', y='DepDelay', data=df2, order=order2, ax=axes[1])
#Q10 AQ has the lowest median on both ArrDelay and DepDelay



