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
%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')
df.info()
df.shape
df.head()
df.set_index('hotel')
df.columns
df.isna().sum()
df.drop(['agent', 'company'], inplace=True, axis=1)
df.isna().sum()
df.dropna(inplace=True)
df.isna().sum()
df.info()
df.isna().sum()
## mode gives the most common values 

df['children'].mode()
df['children'].fillna(0)
df.arrival_date_month.value_counts()
sns.countplot(df.arrival_date_year, hue=df['hotel'])
chart = sns.countplot(df.arrival_date_month)

chart.set_xticklabels(chart.get_xticklabels(), rotation=45,horizontalalignment='right',

    fontweight='light',

    fontsize='x-large')

plt.figure(figsize = (10,5)) ## This will increase the figure size, figsize = (width, height) in inches

chart = sns.countplot(df.arrival_date_day_of_month)

#chart.set_xticklabels(chart.get_xticklabels(), rotation='90', horizontalalignment='right',fontweight='light',

#    fontsize='small')
plt.figure(figsize=(30,5))

df.groupby('country')['country'].size().sort_values(ascending=False).plot.bar()
port_df = df[df['country'] == 'PRT']
plt.figure(figsize = (10,5))

chart = sns.countplot(port_df['arrival_date_month'])

chart.set_xticklabels(chart.get_xticklabels(), rotation='45')
port_df.groupby('arrival_date_month')['arrival_date_month'].count().plot.bar()
port_df.groupby('arrival_date_month')['arrival_date_month'].count()
## diffrent between size and count is, count counts non-na cells vs, size counts all including

## non-na cells

port_df.groupby('arrival_date_month')['arrival_date_month'].size()
plt.figure(figsize=(10,4))

sns.countplot(df.market_segment, hue=df.hotel)
plt.figure(figsize=(10,4))

sns.countplot(df.is_canceled, hue=df.hotel)
df.groupby('hotel')['is_canceled'].sum()
plt.figure(figsize=(10,4))

sns.countplot(df.meal, hue=df.hotel)
## total booking changes for each hotel, sum will sum the cell values

bk_change = df.groupby('hotel')['booking_changes'].sum()



## no booking change, count will do total cells

no_change = df[df['booking_changes'] == 0].groupby('hotel')['hotel'].count()



print("Total Booking Changes Done each", bk_change)

print("\nTotal No Changes Done each", no_change)



print("\n% of total booking changes = ", bk_change/(bk_change + no_change))

df.groupby('hotel')['stays_in_weekend_nights'].count()
df.groupby('hotel')['stays_in_weekend_nights'].sum()
plt.title("Total Stays in weekend nights in each hotel")

df.groupby('hotel')['stays_in_weekend_nights'].sum().plot.bar()
print(df.groupby('hotel')['days_in_waiting_list'].sum())



sns.stripplot(df['hotel'], df['days_in_waiting_list'])
df1 = df[(df['deposit_type'] == 'Non Refund') & (df['is_canceled'] == 1)]

plt.title('Hotel with customer having Non Refund deposit and done cancellations')

sns.countplot(df1['hotel'])
plt.title('Hotel with customer having Non Refund deposit and done cancellations')

sns.countplot(df1['hotel'], hue=df['customer_type'])


df[df['deposit_type'] != "No Deposit"]['deposit_type']
plt.figure(figsize=(10,4))

plt.title('Hotel with customer_type and done cancellations')

df1 = df[df['is_canceled'] > 0]



#sns.countplot(df1['customer_type'], hue=df['is_canceled'])

df1.groupby('customer_type')['is_canceled'].count().plot.bar()
customer = df.groupby(['customer_type','is_canceled'])['customer_type'].count()

print(customer)
plt.figure(figsize=(10,4))

plt.title('Each hotel booking from different distribution channels')

sns.countplot(df.distribution_channel, hue=df.hotel)
df.groupby('customer_type')['required_car_parking_spaces'].sum().plot.bar()
plt.title('Repeated Guest in Each hotel')

df.groupby('hotel')['is_repeated_guest'].sum().plot.bar()