# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
uber_sept = pd.read_csv("/kaggle/input/uber-pickups-in-new-york-city/uber-raw-data-sep14.csv")

uber_sept.head()
uber_sept.shape
uber_sept["Date/Time"] = uber_sept["Date/Time"].map(pd.to_datetime)
uber_sept.head()
def getday(date):

    return date.day

uber_sept["Day"] = uber_sept["Date/Time"].map(getday)

uber_sept.head()
uber_sept.tail()
def getweekday(date):

    return date.weekday()

uber_sept["Weekday"] = uber_sept["Date/Time"].map(getweekday)



def gethour(date):

    return date.hour

uber_sept["Hour"] = uber_sept["Date/Time"].map(gethour)

uber_sept.head()
uber_sept.tail()

from matplotlib import pyplot as plt

import seaborn as sns

plt.hist(uber_sept.Day, bins=30, rwidth=0.9, range=(0.1, 30.1), color="#838383");

plt.grid()

plt.xlabel("Day of the month")

plt.ylabel("Frequency of the rides")

plt.title("Frequency by Day in Sept 2014");
def getCount(rows):

    return len(rows)

by_date = uber_sept.groupby("Day").apply(getCount)

by_date.tail()
plt.plot(by_date);
plt.bar(range(0,30), by_date);

plt.grid()
dat_sort = by_date.sort_values()

dat_sort
plt.figure(figsize=(15,9));

plt.xlabel("Day of the month")

plt.ylabel("Frequency of the rides")

plt.title("Frequency by Day in Sept 2014")

plt.bar(range(1, 31), dat_sort)

plt.grid()

plt.xticks(range(1, 30), dat_sort.index);

by_day = uber_sept.groupby("Weekday").apply(getCount)

by_day.tail()
plt.figure(figsize=(8,6));

plt.xlabel("Day of the week")

plt.ylabel("Frequency of the rides")

plt.title("Frequency by Day in Sept 2014")

plt.xticks(range(7), 'Mon Tue Wed Thur Fri Sat Sun'.split())

plt.bar(range(7), by_day, color='#cf89fa')

plt.grid()

by_hour = uber_sept.groupby("Hour").apply(getCount)

by_hour.tail()

plt.figure(figsize=(12,6));

plt.xlabel("Hour of the Day")

plt.ylabel("Frequency of the rides")

plt.title("Frequency by Day in Sept 2014")

plt.xticks(range(1, 25))

plt.bar(range(1, 25), by_hour, color='#cf87fa')

plt.grid()

cross = uber_sept.groupby("Weekday Hour".split()).apply(getCount).unstack()

cross.tail(10)
sns.set(rc={'figure.figsize':(13,7)})

sns.heatmap(cross);



#sns.yticks(range=(7),'Mon Tue Wed Thur Fri Sat Sun'.split())
plt.figure(figsize=(15,9));

plt.xlabel("Latitude in New York City")

plt.ylabel("Frequency of the rides")

plt.title("Frequency by Day in Sept 2014")

plt.hist(uber_sept["Lat"], bins=100, range=(40.57, 40.89));
plt.figure(figsize=(15,9));

plt.xlabel("Longitude in New York City")

plt.ylabel("Frequency of the rides")

plt.title("Frequency by Day in Sept 2014")

plt.hist(uber_sept["Lon"], bins=100, range=(-74.2, -73.75) );
plt.hist(uber_sept["Lat"], bins=100, range=(40.59, 40.87), color='#232323', alpha=0.5)

plt.twiny()

plt.hist(uber_sept["Lon"], bins=100, range=(-74.2, -73.75), color='#010101', alpha=0.5);
plt.hist(uber_sept["Lat"], bins=100, range=(40.59, 40.87), color='green', alpha=0.6, label='Latitude')

plt.grid()

plt.legend(loc="best")

plt.twiny()

plt.hist(uber_sept["Lon"], bins=100, range=(-74.2, -73.75), color='red', alpha=0.6, label='Longitude')

plt.grid()

plt.legend(loc="upper left");
plt.figure(figsize=(13, 14))

plt.plot(uber_sept["Lat"], uber_sept["Lon"], '.', ms=0.85, alpha=0.7);

plt.xlim(40.4, 41.2)

plt.ylim(-74.5, -73.0);