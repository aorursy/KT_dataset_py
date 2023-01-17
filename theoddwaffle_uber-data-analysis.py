#Kenny Zhu

#Jonathan Xu

#UCSD  Cogs9 Spring 2017



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





DATA_FILE= '../input/uber-raw-data-sep14.csv'

uber_data = pd.read_csv(DATA_FILE)

uber_data.head()

    





# Any results you write to the current directory are saved as output.
uber_data['Date/Time'] = pd.to_datetime(uber_data['Date/Time'], format="%m/%d/%Y %H:%M:%S")

uber_data['DayOfWeekNum'] = uber_data['Date/Time'].dt.dayofweek

uber_data['DayOfWeek'] = uber_data['Date/Time'].dt.weekday_name

uber_data['MonthDayNum'] = uber_data['Date/Time'].dt.day

uber_data['HourOfDay'] = uber_data['Date/Time'].dt.hour

uber_data['MinOfDay'] = uber_data['Date/Time'].dt.minute

uber_data.head()
weekday = uber_data.pivot_table(index=['DayOfWeek'],

                                  values='Base',

                                  aggfunc='count')

weekday.head()

weekdayAverage= weekday/30

weekdayAverage.head()
weekdayAverage.plot(kind='bar')

plt.ylabel('Average Rides Per Day')

plt.title('Average Rides per Day vs Day of Week')
Hours = uber_data.pivot_table(index=['HourOfDay'],

                                  values='Base',

                                  aggfunc='count')

Hours/30

Hours.plot(kind='bar')

plt.ylabel('Number of Rides')

plt.title('Number of Rides vs Hour of Day')
avgHours=Hours/30

avgHours.plot(kind='bar')
min = uber_data.pivot_table(index=['MinOfDay'],

                                  values='Base',

                                  aggfunc='count')

min.plot(kind='bar')

plt.ylim(16500,18000)



min.max()

#minute 10
min.min()

#minute 53
print(Hours)