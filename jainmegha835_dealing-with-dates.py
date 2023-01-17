import numpy as np
import pandas as pd
dates = pd.read_csv("../input/lesson-16-dates/dates_lesson_16.csv")

dates # Check the dates
for col in dates:
    print (type(dates[col][1]))
dates = pd.read_csv("../input/lesson-16-dates/dates_lesson_16.csv", 
                    parse_dates=[0,1,2,3]) # Convert cols to Timestamp
dates
dt=dates['month_day_year']
dt

# Get hour detail from time data 
dates.date_time.dt.hour.head()
import pandas
# Input present datetime using Timestamp 
t = pandas.tslib.Timestamp.now() 
t 

for col in dates:
    print (type(dates[col][1]))
odd_date = "12:30:15 2015-29-11"
pd.to_datetime(odd_date,
               format= "%H:%M:%S %Y-%d-%m") 
date_time=dates['date_time']
[pd.to_datetime(date,format="%H:%M:%S %Y-%d-%m") for date in date_time]
pd.Timestamp('1996-08-11 09:50:35',tz='Europe/London')
dataframe = pd.DataFrame()
dataframe['date'] = pd.date_range('1/1/2001', periods=100000, freq='H')
dataframe[(dataframe['date'] > '2002-1-1 01:00:00') & (dataframe['date'] <= '2002-1-1 04:00:00')]


column_1 = dates.iloc[:,0]

pd.DataFrame({"year": column_1.dt.year,
              "month": column_1.dt.month,
              "day": column_1.dt.day,
              "hour": column_1.dt.hour,
              "dayofyear": column_1.dt.dayofyear,
              "week": column_1.dt.week,
              "weekofyear": column_1.dt.weekofyear,
              "dayofweek": column_1.dt.dayofweek,
              "weekday": column_1.dt.weekday,
              "quarter": column_1.dt.quarter,
             })
print(dates.iloc[1,0])
print(dates.iloc[3,0])
print(dates.iloc[3,0]-dates.iloc[1,0])
dataframe = pd.DataFrame()
dataframe['Arrived']= [pd.Timestamp('01-01-2017'), pd.Timestamp('01-04-2017')] 
dataframe['Left'] = [pd.Timestamp('01-01-2017'), pd.Timestamp('01-06-2017')]
dataframe['Left'] - dataframe['Arrived']

pd.Series(delta.days for delta in (dataframe['Left'] - dataframe['Arrived']))

date=dates['month_day_year']

date.dt.weekday_name


dates['m_date'] = pd.to_datetime(dates['year_month_day'])
start_date = '1996-02-12'
end_date = '2007-09-20'
mask = (dates['m_date'] > start_date) & (dates['m_date'] <= end_date)
mask
df = dates.loc[mask]
df