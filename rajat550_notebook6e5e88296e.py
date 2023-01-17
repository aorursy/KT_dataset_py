import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline
janfeb = pd.read_csv(r'../input/Uber-Jan-Feb-FOIL.csv')

janfeb.head(10)
df_dispatching_base = janfeb.pivot_table(index =['dispatching_base_number'], values='active_vehicles', aggfunc='sum')

df_dispatching_base.plot(kind='bar', figsize =(8,6),color='red')

plt.title('Active vehicles per base')
df_april14 = pd.read_csv(r'../input/uber-raw-data-apr14.csv')

df_april14.head(5)
df_april14.info()
df_april14['Date/Time']= pd.to_datetime(df_april14['Date/Time'], format="%m/%d/%Y %H:%M:%S")

df_april14['DayOfWeekNum']= df_april14['Date/Time'].dt.dayofweek

df_april14['DayOfWeek']= df_april14['Date/Time'].dt.weekday_name

df_april14['MonthDayNum']=df_april14['Date/Time'].dt.day

df_april14['HourOfDay']=df_april14['Date/Time'].dt.hour

df_april14.head(8)
april14_weekdays = df_april14.pivot_table(index=['MonthDayNum'],values='Base',aggfunc='count')

april14_weekdays.plot(kind='bar', figsize=(8,6), color='brown')

plt.ylabel('Total Journeys')

plt.title('Journeys by Month Day')
april14_weekdays = df_april14.pivot_table(index=['DayOfWeekNum','DayOfWeek'],values='Base',aggfunc='count')

april14_weekdays.plot(kind='bar', figsize=(8,6),color='brown')

plt.ylabel('Total Journeys')

plt.title('Journeys by Week Day')
april14_weekdays = df_april14.pivot_table(index=['HourOfDay'],values='Base',aggfunc='count')

april14_weekdays.plot(kind='bar', figsize=(8,6), color='yellow')

plt.ylabel('Total Journeys')

plt.title('Journeys by Hour of Day')