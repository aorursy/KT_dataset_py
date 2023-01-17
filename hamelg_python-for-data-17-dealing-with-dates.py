import numpy as np

import pandas as pd
dates = pd.read_csv("../input/lesson-16-dates/dates_lesson_16.csv")



dates # Check the dates
for col in dates:

    print (type(dates[col][1]))
dates = pd.read_csv("../input/lesson-16-dates/dates_lesson_16.csv", 

                    parse_dates=[0,1,2,3]) # Convert cols to Timestamp
for col in dates:

    print (type(dates[col][1]))
odd_date = "12:30:15 2015-29-11"
pd.to_datetime(odd_date,

               format= "%H:%M:%S %Y-%d-%m") 
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