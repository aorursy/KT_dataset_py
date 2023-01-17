# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# import data

hotel = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv', parse_dates = True)
hotel.head()
hotel.shape
hotel.info()
hotel.describe()
# list of categorical data

category_list = ['hotel','is_canceled', 'meal', 'country','market_segment', 'distribution_channel', 'is_repeated_guest',\

                 'reserved_room_type','assigned_room_type','deposit_type', 'agent','company','customer_type','reservation_status']
for col in category_list:

    print(hotel[col].value_counts())
# convert type to categorical

for col in category_list:

    hotel[col] = hotel[col].astype('category')
hotel.info()
# deal with date

from datetime import datetime

from datetime import timedelta

import calendar
dict_month = {j:i for i,j in enumerate(calendar.month_name)}

dict_month
hotel['arrival_date_month'].value_counts()
hotel['arrival_date_month'] = hotel['arrival_date_month'].map(dict_month)
hotel['arrival_date_month'].value_counts()
# create arrival_date

hotel['arrival_date'] = hotel.arrival_date_year.astype(str)+'-'+hotel.arrival_date_month.astype(str)+"-"+hotel.arrival_date_day_of_month.astype(str)
hotel['arrival_date'] = pd.to_datetime(hotel['arrival_date'])

hotel['arrival_date'].head()
hotel['reservation_status_date'] = pd.to_datetime(hotel['reservation_status_date'])
hotel.info()
import matplotlib.pyplot as plt

import seaborn as sns
numeric_list = hotel.columns[hotel.dtypes == np.float64].append(hotel.columns[hotel.dtypes == np.int64])
for col in numeric_list:

    hotel[col].plot.box()

    plt.title(col)

    plt.show()
# adr change with respect to arrival date

plt.figure(figsize = (10,10))

sns.lineplot(data = hotel, x = 'arrival_date_month', y = 'adr', hue = 'arrival_date_year', palette = 'muted')

plt.figure(figsize=(20,30))

plt.subplot(3,1,1)

sns.lineplot(x="arrival_date_day_of_month", y="adr",hue="arrival_date_month", \

             data=hotel[hotel.arrival_date_year == 2015],palette = 'muted',estimator = None)

plt.subplot(3,1,2)

sns.lineplot(x="arrival_date_day_of_month", y="adr",hue="arrival_date_month", \

             data=hotel[hotel.arrival_date_year == 2016],palette = 'muted',estimator = None)

plt.subplot(3,1,3)

sns.lineplot(x="arrival_date_day_of_month", y="adr",hue="arrival_date_month", \

             data=hotel[hotel.arrival_date_year == 2017],palette = 'muted',estimator = None)
# extremely large value, may be an error

hotel.iloc[hotel.adr.idxmax()]
# remove this observation

hotel1 = hotel.drop(hotel.adr.idxmax(), axis = 0)
hotel1.shape
numeric_list
# exclude adr and dates

num_list = numeric_list[[0,2]].append(numeric_list[7:17])
for var in num_list:

    plt.figure(figsize = (10,10))

    sns.scatterplot( x = var, y = 'adr', data = hotel1, hue = 'hotel', alpha = 0.2, style = 'hotel')
# fill in missing values of children using mode

hotel1.children.value_counts()
hotel1.children.fillna(0, inplace = True)

hotel1.children.value_counts()
hotel1.isnull().sum()
# variables like adults, children, babies, better use boxplot to plot against adr

plt.figure(figsize = (10,10))

sns.boxplot(x = 'adults', y = 'adr', data = hotel1)
# adr for adults>=5 are all zero because all canceled

hotel1[hotel1['adults']>=5].is_canceled
plt.figure(figsize = (10,10))

sns.boxplot(x = 'children', y = 'adr', data = hotel1)
hotel1[hotel1['children']==10]
plt.figure(figsize = (10,10))

sns.boxplot(x = 'babies', y = 'adr', data = hotel1)

plt.show()
corr = hotel[numeric_list].corr()
# heatmap

plt.figure(figsize = (10,10))

sns.heatmap(corr,cmap="YlGnBu")

plt.show()
for col in category_list:

    plt.figure(figsize = (6,6))

    sns.boxplot(x = col, y = 'adr', data = hotel1)

    plt.xticks(rotation = 60)

    plt.show()