# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dfc = pd.read_csv('../input/calendar.csv')

dfl = pd.read_csv('../input/listings.csv')

dfc.head(10)

# Check start and end period to determine the sample of research

dfc.min()
dfc.max()
#Check whether is there any listing room who has two condition (available and non available) in the same day

dfcek = dfc.groupby(['date','listing_id']).count()

dfcek[dfcek['available']>1]
# check whether is there any change of total quantity listing room in the sample period

dfc.groupby('date').count().head(5)
dfc.groupby('date').count().tail(5)
dfc.listing_id.nunique()
#prepare data

dfc['date'] = pd.to_datetime(dfc['date'])

dfc1 = dfc[dfc['available']== 'f']

dfc2 = dfc1.groupby('date')['listing_id'].count().reset_index()

dfc2['date'] = pd.to_datetime(dfc2['date'])

dfc2['Month'] = dfc2['date'].dt.strftime('%b')

dfc2['Day'] = dfc2['date'].dt.day   

dfc2.set_index(['Month','Day']).sum(level=[0,1]).reset_index()

dfc2[dfc2['Month']=='Jan']

#design a visualization. Make a chart (a whole data to know about the trend of data)

plt.figure(figsize=(15, 8))

plt.plot(dfc2.date, dfc2.listing_id, color='b', marker='*', linewidth=0.3)

plt.title("Count of Booking Room Demand")

plt.xlabel('Date')

plt.ylabel('Count listing_id')

#Prepare data for cluster based on period.

dfc2['date'] = pd.to_datetime(dfc2['date'])

dfc2["weekday"] = dfc2["date"].dt.weekday_name

dfc3 = dfc2[(dfc2['date']>='2016-01-01')&(dfc2['date']<='2016-04-01')]

dfc4 = dfc2[(dfc2['date']>'2016-04-01')&(dfc2['date']<'2016-07-01')]

dfc5 = dfc2[(dfc2['date']>='2016-07-01')&(dfc2['date']<='2017-01-02')]

dfc2.head(5)

#all data

plt.figure(figsize=(15, 8))

plt.scatter(dfc2.weekday, dfc2.listing_id, color='b', marker='.')

plt.title("Average listing price by date")

plt.xlabel('date')

plt.ylabel('average listing price')

plt.grid()

plt.figure(figsize=(10, 6))

sns.boxplot(x = 'weekday',  y = 'listing_id', data = dfc2, palette="Blues", width=0.4)

plt.title("Count of Booking Room Demand Weekly Analysis")

plt.xlabel('Name of Day')

plt.ylabel('Count of Listing_id')

plt.show()
plt.figure(figsize=(10, 6))

sns.boxplot(x = 'weekday',  y = 'listing_id', data = dfc3, palette="Blues", width=0.4)

plt.title("Count of Booking Room Demand Weekly Analysis Cluster I")

plt.xlabel('Name of Day')

plt.ylabel('Count of Listing_id')

plt.show()
plt.figure(figsize=(10, 6))

sns.boxplot(x = 'weekday',  y = 'listing_id', data = dfc4, palette="Blues", width=0.4)

plt.title("Count of Booking Room Demand Weekly Analysis Cluster II")

plt.xlabel('Name of Day')

plt.ylabel('Count of Listing_id')

plt.show()
plt.figure(figsize=(10, 6))

sns.boxplot(x = 'weekday',  y = 'listing_id', data = dfc5, palette="Blues", width=0.4)

plt.title("Count of Booking Room Demand Weekly Analysis Cluster III")

plt.xlabel('Name of Day')

plt.ylabel('Count of Listing_id')

plt.show()
#do montlhy analysis. Create visualization in monthly.

plt.figure(figsize=(15, 8))

sns.boxplot(x = 'Day',  y = 'listing_id', data = dfc2, palette="Purples", width=0.4)

plt.title("Count of Booking Room Demand Monthly Analysis")

plt.xlabel('Name of Day')

plt.ylabel('Count of Listing_id')

plt.show()


dfc7 = dfc2[dfc2['listing_id']<1500]

plt.figure(figsize=(15, 8))

sns.boxplot(x = 'Day',  y = 'listing_id', data = dfc7, palette="Purples", width=0.4)

plt.title("Count of Booking Room Demand Monthly Analysis")

plt.xlabel('Name of Day')

plt.ylabel('Count of Listing_id')

plt.show()

#cluster I

plt.figure(figsize=(15, 8))

sns.boxplot(x = 'Day',  y = 'listing_id', data = dfc3, palette="Purples", width=0.4)

plt.title("Count of Booking Room Demand Monthly Analysis Cluster I")

plt.xlabel('Name of Day')

plt.ylabel('Count of Listing_id')

plt.show()
#Cluster II

plt.figure(figsize=(15, 8))

sns.boxplot(x = 'Day',  y = 'listing_id', data = dfc4, palette="Purples", width=0.4)

plt.title("Count of Booking Room Demand Monthly Analysis Cluster II")

plt.xlabel('Name of Day')

plt.ylabel('Count of Listing_id')

plt.show()
dfc7 = dfc2[dfc2['listing_id']<1500]

plt.figure(figsize=(15, 8))

sns.boxplot(x = 'Day',  y = 'listing_id', data = dfc5, palette="Purples", width=0.4)

plt.title("Count of Booking Room Demand Monthly Analysis Cluster III")

plt.xlabel('Name of Day')

plt.ylabel('Count of Listing_id')

plt.show()