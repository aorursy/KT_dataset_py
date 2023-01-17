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
# Importing libraries necessary for the study

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# reading the dataset

hotel_df = pd.DataFrame(pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv'))

hotel_df.head() 
hotel_df.shape
hotel_df.info()
hotel_df.describe()
round(100*(hotel_df.isnull().sum()/len(hotel_df.index)), 2).sort_values(ascending = False) 
#Removing columns having missing value % is >=50

hotel_df.drop('company',axis=1,inplace=True)
hotel_df.dtypes
#Remove null values 

hotel_df.dropna(inplace=True)
round(100*(hotel_df.isnull().sum()/len(hotel_df.index)), 2).sort_values(ascending = False) 
#unique value in dataframe

cols = hotel_df.columns

for i in cols:

    print(i,hotel_df[i].unique(),'\n','*********************************************')
#total number of guests

hotel_df['guests'] = hotel_df['adults'] + hotel_df['children'] + hotel_df['babies'] 
hotel_df['rate_per_person'] = hotel_df['adr'] / (hotel_df['adults'] + hotel_df['children']  )

hotel_df.rate_per_person.round(2)
hotel_df.dtypes
sns.countplot(y=hotel_df['hotel'])
sns.countplot(y=hotel_df['customer_type'])
sns.countplot(y=hotel_df['meal'],order=hotel_df['meal'].value_counts().index)
sns.countplot(y=hotel_df['market_segment'],order = hotel_df['market_segment'].value_counts().index)
sns.countplot(y=hotel_df['arrival_date_month'],order = hotel_df['arrival_date_month'].value_counts().index)
sns.countplot(y=hotel_df['distribution_channel'],order = hotel_df['distribution_channel'].value_counts().index)
sns.countplot(y=hotel_df['deposit_type'],order = hotel_df['deposit_type'].value_counts().index)
sns.countplot(y=hotel_df['reservation_status'],order = hotel_df['reservation_status'].value_counts().index)
sns.barplot(x="hotel", y="lead_time", hue="is_canceled", data=hotel_df.groupby(["hotel","is_canceled"]).lead_time.count().reset_index())

plt.ylabel('count of lead_time')

plt.show()
plt.figure(figsize=(16, 6))

sns.barplot(x="distribution_channel", y="is_canceled", hue="hotel", data=hotel_df)

plt.ylabel('is_canceled count')

plt.show()
plt.figure(figsize=(16, 6))

sns.barplot(x="market_segment", y="is_canceled", hue="hotel", data=hotel_df)

plt.ylabel('is_canceled count')

plt.show()
plt.figure(figsize=(16, 6))

sns.barplot(y="total_of_special_requests", x="reserved_room_type", hue="hotel", data=hotel_df)

plt.ylabel('special request count')

plt.show()
plt.figure(figsize=(12, 8))

sns.lineplot(x = "arrival_date_month", y="guests", hue="reservation_status", data=hotel_df,  sizes=(2.5, 2.5),sort=False)
plt.figure(figsize=(12, 8))

sns.lineplot(x = "arrival_date_month", y="guests", hue="hotel", data=hotel_df, size="hotel", sizes=(2.5, 2.5),sort=False)
plt.figure(figsize=(12, 8))

sns.lineplot(x = "arrival_date_month", y="total_of_special_requests", hue="hotel", data=hotel_df, size="hotel", sizes=(2.5, 2.5),sort=False)
plt.figure(figsize=(12, 8))

sns.lineplot(x = "arrival_date_month", y="booking_changes", hue="hotel", data=hotel_df, size="hotel", sizes=(2.5, 2.5),sort=False)
plt.figure(figsize=(12, 8))

sns.lineplot(x = "arrival_date_month", y="is_repeated_guest", hue="hotel", data=hotel_df, size="hotel", sizes=(2.5, 2.5),sort=False)
plt.figure(figsize=(12, 8))

sns.lineplot(x = "arrival_date_month", y="previous_cancellations", hue="hotel", data=hotel_df, size="hotel", sizes=(2.5, 2.5),sort=False)
plt.figure(figsize=(12, 8))

sns.lineplot(x = "arrival_date_month", y="previous_bookings_not_canceled", hue="hotel", data=hotel_df, size="hotel", sizes=(2.5, 2.5),sort=False)
plt.figure(figsize=(12, 8))

sns.lineplot(x = "arrival_date_month", y="stays_in_weekend_nights", hue="hotel", data=hotel_df, size="hotel", sizes=(2.5, 2.5),sort=False)
plt.figure(figsize=(12, 8))

sns.lineplot(x = "arrival_date_month", y="stays_in_week_nights", hue="hotel", data=hotel_df, size="hotel", sizes=(2.5, 2.5),sort=False)
plt.figure(figsize=(16, 6))

sns.barplot(x="assigned_room_type", y="rate_per_person", hue="hotel", data=hotel_df)

plt.ylabel('rate_per_person count')

plt.show()
plt.figure(figsize=(12, 8))

sns.lineplot(x = "arrival_date_month", y="booking_changes", hue="hotel", data=hotel_df, size="hotel", sizes=(2.5, 2.5),sort=False)
#heatmap

plt.figure(figsize=(25,25))

sns.heatmap(hotel_df.corr(), annot= True)