# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

from matplotlib import cm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
DATA_FILE = '../input/dataset-v2-coma/z14v2coma.csv'

pizza_data = pd.read_csv(DATA_FILE)

pizza_data.head()
pizza_data.tail()
pizza_data.info()
pizza_data.describe()
pizza_data['interval'] = pd.to_datetime(pizza_data['interval'], format="%d/%m/%Y %H:%M")

pizza_data['DayOfWeekNum'] = pizza_data['interval'].dt.dayofweek

pizza_data['DayOfWeek'] = pizza_data['interval'].dt.weekday_name

pizza_data['MonthDayNum'] = pizza_data['interval'].dt.day

pizza_data['HourOfDay'] = pizza_data['interval'].dt.hour
pizza_views_weekdays = pizza_data.pivot_table(index=['DayOfWeekNum','DayOfWeek'],

                                  values='views',

                                  aggfunc='count')

pizza_views_weekdays.plot(kind='bar', figsize=(20,18), color='orange')

plt.ylabel('views')

plt.title('views by Week Day');



pizza_impressions_weekdays = pizza_data.pivot_table(index=['DayOfWeekNum','DayOfWeek'],

                                  values='impressions',

                                  aggfunc='count')

pizza_impressions_weekdays.plot(kind='bar', figsize=(20,18), color= 'gray')

plt.ylabel('impressions')

plt.title('impressions by Week Day');
pizza_views_monthdays = pizza_data.pivot_table(index=['MonthDayNum'],

                                  values='views',

                                  aggfunc='count')

pizza_monthdays.plot(kind='bar', figsize=(8,6))

plt.ylabel('Total views')

plt.title('views by Month Day');
pizza_views_hour = pizza_data.pivot_table(index=['HourOfDay'],

                                  values='views',

                                  aggfunc='count')

pizza_views_hour.plot(kind='bar', figsize=(8,6))

plt.ylabel('Total views')

plt.title('views by Hour');



pizza_impressions_hour = pizza_data.pivot_table(index=['HourOfDay'],

                                  values='impressions',

                                  aggfunc='count')

pizza_impressions_hour.plot(kind='bar', figsize=(8,6))

plt.ylabel('Total impressions')

plt.title('impressions by Hour');
plt.rcParams['figure.figsize']=(20,10) # set the figure size

plt.style.use('fivethirtyeight') # using the fivethirtyeight matplotlib theme

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()  # set up the 2nd axis

ax1.plot(pizza_views_hour, color="blue", linewidth=4.5, linestyle="-") #plot the Revenue on axis #1

ax2.plot(pizza_impressions_hour, color="red",  linewidth=2.5, linestyle="--")

plt.show()