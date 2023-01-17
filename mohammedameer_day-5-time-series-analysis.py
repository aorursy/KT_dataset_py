import datetime



# Present Date and TIme

date_time_now = datetime.datetime.now()

date_time_now
print(date_time_now)
# Present Date only

date = datetime.date.today()

print(date)
# Only the time to be printed

print(date_time_now.now().time())
# List of classes in datetime module

dir(datetime)
# Manually declaring the complete date

print(datetime.date(2020,8,20))
print(date)
# Splitting the dates

print(f'Year = {date.year}')

print(f'Month = {date.month}')

print(f'Day = {date.day}')
from datetime import time



a = time(11,29,45)



print(a)

print(type(a))
print(f'Hour = {a.hour}')

print(f'Minute = {a.minute}')

print(f'second = {a.second}')
demo_date1 = '20/11/1995'

demo_date1
type(demo_date1)
import pandas as pd

pd.to_datetime(demo_date1)
demo_date2 = '11/2020/03 23:23'

pd.to_datetime(demo_date2,format = '%d/%Y/%m %H:%M')
import pandas as pd

df = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')

df.head()
df.info()
df['DATE_TIME'][0]
# COnvert the DATE_TIME column from String datatype to Datetime datatype

df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'],format = '%d-%m-%Y %H:%M')

df.info()
df
# Splitting DateTime column into Date column and Time column

df['DATE'] = pd.to_datetime(df['DATE_TIME'],format = '%d-%m-%Y %H:%M').dt.date
df['TIME'] = pd.to_datetime(df['DATE_TIME'],format = '%d-%m-%Y %H:%M').dt.time
df.info()
df['DATE'] = pd.to_datetime(df['DATE'])
df.info()
df['TIME'] = pd.to_datetime(df['TIME'],format = '%H:%M:%S')
df.info()
df
df['DATE'].nunique()



# This dataset is for totally 34 days
# Starting Date

df['DATE'].min()
df['DATE_TIME'].min()
# Ending Date

df['DATE'].max()
df['DATE_TIME'].max()
date_unique = df['DATE'].value_counts()

date_unique
df['SOURCE_KEY'].nunique()
df['TIME'].unique()
# 22 inverters

# 4 times per hour

# 24 hours

22*4*24
date_unique

date_unique.sort_index()
import matplotlib.pyplot as plt

plt.figure(figsize=(12,7))

plt.bar(date_unique.index,date_unique)

plt.xticks(ticks = date_unique.index,rotation = 90)

plt.show()
df2 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')

df2.head()
df2['DATE_TIME'] = pd.to_datetime(df2['DATE_TIME'],format = '%Y-%m-%d %H:%M')
df2['DATE'] = pd.to_datetime(df2['DATE_TIME'],format = '%Y-%m-%d %H:%M').dt.date
df2['DATE'] = pd.to_datetime(df2['DATE'])
df2
d = df2.groupby(['DATE']).sum()

d['IRRADIATION']
x = []