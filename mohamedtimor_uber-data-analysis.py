import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import cm
uber_data = pd.read_csv('../input/uber-pickups-in-new-york-city/uber-raw-data-aug14.csv')

uber_data.head()
uber_data.info()
uber_data['Date/Time'] = pd.to_datetime(uber_data['Date/Time'], format="%m/%d/%Y %H:%M:%S")

uber_data['DayOfWeekNum'] = uber_data['Date/Time'].dt.dayofweek

uber_data['DayOfWeek'] = uber_data['Date/Time'].dt.weekday

uber_data['MonthDayNum'] = uber_data['Date/Time'].dt.day

uber_data['HourOfDay'] = uber_data['Date/Time'].dt.hour
uber_data.head()
uber_weekdays = uber_data.pivot_table(index=['DayOfWeekNum','DayOfWeek'],

                                  values='Base',

                                  aggfunc='count')

uber_weekdays.plot(kind='bar', figsize=(8,6))

plt.ylabel('Total Journeys')

plt.title('Journeys by Week Day')
uber_monthdays = uber_data.pivot_table(index=['MonthDayNum'],

                                  values='Base',

                                  aggfunc='count')

uber_monthdays.plot(kind='bar', figsize=(8,6))

plt.ylabel('Total Journeys')

plt.title('Journeys by Month Day');
uber_hour = uber_data.pivot_table(index=['HourOfDay'],

                                  values='Base',

                                  aggfunc='count')

uber_hour.plot(kind='bar', figsize=(8,6))

plt.ylabel('Total Journeys')

plt.title('Journeys by Hour');