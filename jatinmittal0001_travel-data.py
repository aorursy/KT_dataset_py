import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("../input/ixigo-practice/analytics_test.csv",sep='\t',)

print('Shape of raw data: ',data.shape)

print(data.columns)
data.head()
from datetime import datetime

data['bookingDate'] = data['bookingDate'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

data['onwardDeparture'] = data['onwardDeparture'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

data['booking_month'] = data['bookingDate'].apply(lambda x: x.month)

data['departure_month'] = data['onwardDeparture'].apply(lambda x: x.month)

data.head()
print(data.booking_month.unique())

print(data.departure_month.unique())
data['orig_dest_pair'] = data['originCode'] + " " + data['destinationCode']

print('Unique pairs are: ',data['orig_dest_pair'].unique())
b = data[data.userType=='newUser'].groupby(['orig_dest_pair','booking_month']).aggregate({'userType':'count'})

b.reset_index(inplace=True,drop=False)

b.rename(columns = {'userType':'num_new_user_bookings'},inplace=True)

b
a = data[data.booking_month == 6].groupby(['orig_dest_pair','booking_month']).aggregate({'userType':'count'})

a.reset_index(inplace=True,drop=False)

a.rename(columns = {'userType':'num_all_user_bookings'},inplace=True)

a
merged_data = a.merge(b,on=['orig_dest_pair','booking_month'])

merged_data['perc_of_new_users'] = round(merged_data['num_new_user_bookings']/merged_data['num_all_user_bookings']*100,2)

merged_data.head()
sns.barplot(y='perc_of_new_users',x='orig_dest_pair',data=merged_data)
print('Different device types')

list(set(data.devicePlatform))
num_users_web = data[(data.devicePlatform=='ixiweb')].groupby(['userId']).aggregate({'bookingId':'count'})

print('Number of users with >2 bookings on web: ',sum((num_users_web.bookingId>2)==True))
def adjust_advance_month(book_month,dep_month):

    a = dep_month - book_month

    if dep_month < book_month:

        a = dep_month + (12-book_month)

    return a
filter_d = data[(data.bookingDate<data.onwardDeparture)][['bookingId','booking_month','departure_month']]

filter_d['months_advance'] = filter_d.apply(lambda x:adjust_advance_month(x.booking_month,x.departure_month),axis=1)

a = filter_d.groupby('months_advance').aggregate({'bookingId':'count'})

a.reset_index(inplace=True,drop=False)

a.rename(columns = {'bookingId':'num_bookings'},inplace=True)

sns.barplot(y='num_bookings', x='months_advance',  data=a[a.months_advance>0])

num_users_web = data[(data['devicePlatform']=='iximaad') | (data['devicePlatform']=='iximaio')].groupby(['userId']).aggregate({'bookingId':'count'})

print('Number of user bookings on Andoid orr iOS: ',sum((num_users_web.bookingId)==True))
print('Number of users reactivated their accounts: ',

      data[data['userType']=='Reactivated'].groupby(['booking_month']).aggregate({'userType':'count'}))

data[data['isInternational']].shape