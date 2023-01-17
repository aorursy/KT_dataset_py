%matplotlib inline
import pandas as pd


df= pd.read_csv('../input/humidity.csv')
df2 = pd.read_csv('../input/pressure.csv')
df3 = pd.read_csv('../input/temperature.csv')
df4 = pd.read_csv('../input/weather_description.csv')
df5 = pd.read_csv('../input/wind_direction.csv')
df6 = pd.read_csv('../input/wind_speed.csv')
merge = pd.merge(df, df2, suffixes=('_humidity', '_pressure'),left_index=True, right_index=True, how='outer')
merge2 = pd.merge(df3, df4, suffixes=('_temp', '_weathercondition'),left_index=True, right_index=True, how='outer' )
merge3=pd.merge(df5, df6, suffixes=('_winddirection', '_windspeed'),left_index=True, right_index=True, how='outer')
merge4 = pd.merge(merge, merge2,left_index=True, right_index=True, how='outer')
merge5 = pd.merge(merge4, merge3,left_index=True, right_index=True, how='outer')

merged = pd.DataFrame(merge5)
merged.head()
merged = merged.reindex(sorted(merged.columns), axis=1)
merged.drop(['datetime_pressure', 'datetime_temp', 'datetime_weathercondition',
            'datetime_winddirection', 'datetime_windspeed'], axis=1, inplace=True)
merged.head()
temp = pd.DatetimeIndex(merged['datetime_humidity'])
merged['date'] = temp.date
merged['year'] = temp.year
merged['month'] = temp.month
merged['day'] = temp.day
merged['time'] = temp.time

del merged['datetime_humidity']

merged = merged[['date', 'year', 'month', 'day', 'time', 'Albuquerque_humidity', 'Albuquerque_pressure', 'Albuquerque_temp',
                 'Albuquerque_weathercondition', 'Albuquerque_winddirection', 'Albuquerque_windspeed',
                 'Atlanta_humidity', 'Atlanta_pressure', 'Atlanta_temp', 'Atlanta_weathercondition',
                 'Atlanta_winddirection', 'Atlanta_windspeed', 'Beersheba_humidity', 'Beersheba_pressure',
                 'Beersheba_temp', 'Beersheba_weathercondition', 'Beersheba_winddirection',
                 'Beersheba_windspeed', 'Boston_humidity', 'Boston_pressure', 'Boston_temp', 'Boston_weathercondition',
                 'Boston_winddirection', 'Boston_windspeed', 'Charlotte_humidity', 'Charlotte_pressure',
                 'Charlotte_temp', 'Charlotte_weathercondition', 'Charlotte_winddirection', 'Charlotte_windspeed',
                 'Chicago_humidity', 'Chicago_pressure', 'Chicago_temp', 'Chicago_weathercondition',
                 'Chicago_winddirection', 'Chicago_windspeed', 'Dallas_humidity', 'Dallas_pressure', 'Dallas_temp',
                 'Dallas_weathercondition', 'Dallas_winddirection', 'Dallas_windspeed', 'Denver_humidity',
                 'Denver_pressure', 'Denver_temp', 'Denver_weathercondition', 'Denver_winddirection',
                 'Denver_windspeed', 'Detroit_humidity', 'Detroit_pressure', 'Detroit_temp',
                 'Detroit_weathercondition', 'Detroit_winddirection', 'Detroit_windspeed', 'Eilat_humidity',
                 'Eilat_pressure', 'Eilat_temp', 'Eilat_weathercondition', 'Eilat_winddirection', 'Eilat_windspeed',
                 'Haifa_humidity', 'Haifa_pressure', 'Haifa_temp', 'Haifa_weathercondition', 'Haifa_winddirection',
                 'Haifa_windspeed', 'Houston_humidity', 'Houston_pressure', 'Houston_temp', 'Houston_weathercondition',
                 'Houston_winddirection', 'Houston_windspeed', 'Indianapolis_humidity', 'Indianapolis_pressure',
                 'Indianapolis_temp', 'Indianapolis_weathercondition', 'Indianapolis_winddirection',
                 'Indianapolis_windspeed', 'Jacksonville_humidity', 'Jacksonville_pressure', 'Jacksonville_temp',
                 'Jacksonville_weathercondition', 'Jacksonville_winddirection', 'Jacksonville_windspeed',
                 'Jerusalem_humidity', 'Jerusalem_pressure', 'Jerusalem_temp', 'Jerusalem_weathercondition',
                 'Jerusalem_winddirection', 'Jerusalem_windspeed', 'Kansas City_humidity', 'Kansas City_pressure',
                 'Kansas City_temp', 'Kansas City_weathercondition', 'Kansas City_winddirection',
                 'Kansas City_windspeed', 'Las Vegas_humidity', 'Las Vegas_pressure', 'Las Vegas_temp',
                 'Las Vegas_weathercondition', 'Las Vegas_winddirection', 'Las Vegas_windspeed', 'Los Angeles_humidity',
                 'Los Angeles_pressure', 'Los Angeles_temp', 'Los Angeles_weathercondition', 'Los Angeles_winddirection',
                 'Los Angeles_windspeed', 'Miami_humidity', 'Miami_pressure', 'Miami_temp', 'Miami_weathercondition',
                 'Miami_winddirection', 'Miami_windspeed', 'Minneapolis_humidity', 'Minneapolis_pressure',
                 'Minneapolis_temp', 'Minneapolis_weathercondition', 'Minneapolis_winddirection',
                 'Minneapolis_windspeed', 'Montreal_humidity', 'Montreal_pressure', 'Montreal_temp',
                 'Montreal_weathercondition', 'Montreal_winddirection', 'Montreal_windspeed', 'Nahariyya_humidity',
                 'Nahariyya_pressure', 'Nahariyya_temp', 'Nahariyya_weathercondition', 'Nahariyya_winddirection',
                 'Nahariyya_windspeed', 'Nashville_humidity', 'Nashville_pressure', 'Nashville_temp',
                 'Nashville_weathercondition', 'Nashville_winddirection', 'Nashville_windspeed', 'New York_humidity',
                 'New York_pressure', 'New York_temp', 'New York_weathercondition', 'New York_winddirection',
                 'New York_windspeed', 'Philadelphia_humidity', 'Philadelphia_pressure', 'Philadelphia_temp',
                 'Philadelphia_weathercondition', 'Philadelphia_winddirection', 'Philadelphia_windspeed',
                 'Phoenix_humidity', 'Phoenix_pressure', 'Phoenix_temp', 'Phoenix_weathercondition',
                 'Phoenix_winddirection', 'Phoenix_windspeed', 'Pittsburgh_humidity', 'Pittsburgh_pressure',
                 'Pittsburgh_temp', 'Pittsburgh_weathercondition', 'Pittsburgh_winddirection', 'Pittsburgh_windspeed',
                 'Portland_humidity', 'Portland_pressure', 'Portland_temp', 'Portland_weathercondition',
                 'Portland_winddirection', 'Portland_windspeed', 'Saint Louis_humidity', 'Saint Louis_pressure',
                 'Saint Louis_temp', 'Saint Louis_weathercondition', 'Saint Louis_winddirection',
                 'Saint Louis_windspeed', 'San Antonio_humidity', 'San Antonio_pressure', 'San Antonio_temp',
                 'San Antonio_weathercondition', 'San Antonio_winddirection', 'San Antonio_windspeed',
                 'San Diego_humidity', 'San Diego_pressure', 'San Diego_temp', 'San Diego_weathercondition',
                 'San Diego_winddirection', 'San Diego_windspeed', 'San Francisco_humidity', 'San Francisco_pressure',
                 'San Francisco_temp', 'San Francisco_weathercondition', 'San Francisco_winddirection',
                 'San Francisco_windspeed', 'Seattle_humidity', 'Seattle_pressure', 'Seattle_temp',
                 'Seattle_weathercondition', 'Seattle_winddirection', 'Seattle_windspeed', 'Tel Aviv District_humidity',
                 'Tel Aviv District_pressure', 'Tel Aviv District_temp', 'Tel Aviv District_weathercondition',
                 'Tel Aviv District_winddirection', 'Tel Aviv District_windspeed', 'Toronto_humidity',
                 'Toronto_pressure', 'Toronto_temp', 'Toronto_weathercondition', 'Toronto_winddirection',
                 'Toronto_windspeed', 'Vancouver_humidity', 'Vancouver_pressure', 'Vancouver_temp',
                 'Vancouver_weathercondition', 'Vancouver_winddirection', 'Vancouver_windspeed']]

merged.head()
datetime = merged[['date', 'year', 'month', 'day' ,'time']]
portland = merged[merged.columns[pd.Series(merged.columns).str.startswith('Portland')]]
pdxmerged =  pd.merge(datetime, portland, left_index=True, right_index=True, how='outer')
pdxmerged.head()
meantemp = pdxmerged.groupby(['date'])['Portland_temp'].mean()
convertedtemp = 1.8*(meantemp-273.15)+32
maxtemp = pdxmerged.groupby(['date'])['Portland_temp'].max()
convertedmaxtemp = 1.8*(maxtemp-273.15)+32
mintemp = pdxmerged.groupby(['date'])['Portland_temp'].min()
convertedmintemp = 1.8*(mintemp-273.15)+32
meanpressure = pdxmerged.groupby(['date'])['Portland_pressure'].mean()
meanwind = pdxmerged.groupby(['date'])['Portland_windspeed'].mean()

meanpdx = pd.DataFrame()
meanpdx['average temp'] = convertedtemp
meanpdx['max temp'] = convertedmaxtemp
meanpdx['min temp'] = convertedmintemp
meanpdx['average wind speed'] = meanwind
meanpdx['average pressure'] = meanpressure

meanpdx.reset_index(level=0, inplace=True)
theyear = pd.DatetimeIndex(meanpdx['date'])
meanpdx['year'] = theyear.year
meanpdx['month'] = theyear.month
meanpdx['day'] = theyear.day

meanpdx.head()
november = pd.DataFrame()
november = meanpdx.loc[meanpdx['month'] == 11]
november.head(50)
novyr12 = pd.DataFrame(november[november['year'] == 2012])
novyr13 = pd.DataFrame(november[november['year'] == 2013])
novyr14 = pd.DataFrame(november[november['year'] == 2014])
novyr15 = pd.DataFrame(november[november['year'] == 2015])
novyr16 = pd.DataFrame(november[november['year'] == 2016])
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np


fig, ax = plt.subplots(nrows=5, ncols=1, figsize = (30,30))
fig.suptitle('Average Temp for November', fontsize=40, x = .5, y=1.05)
fig.tight_layout() 

ax = plt.subplot(511)
ax.set_title("Nov 2012", fontsize=30)
plt.plot(novyr12['date'], novyr12['average temp'])
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.margins(0.09)
plt.grid()

plt.subplot(512)
ax.set_title("Nov 2013", fontsize=30)
plt.plot(novyr13['date'], novyr13['average temp'])
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.margins(0.09)
plt.grid()

plt.subplot(513)
ax.set_title("Nov 2014", fontsize=30)
plt.plot(novyr14['date'], novyr14['average temp'])
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.margins(0.09)
plt.grid()

plt.subplot(514)
ax.set_title("Nov 2015", fontsize=30)
plt.plot(novyr15['date'], novyr15['average temp'])
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.margins(0.09)
plt.grid()

plt.subplot(515)
ax.set_title("Nov 2016", fontsize=30)
plt.plot(novyr16['date'], novyr16['average temp'])
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.margins(0.09)
plt.grid()

