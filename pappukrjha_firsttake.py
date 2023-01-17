# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
print('No of rows %d and No of Cols %d in the Data' %(train.shape[0], train.shape[1]))

print('Data View!!')

print(train.head(3))

print('\nData Type!!\n')

print(train.dtypes)

print('\n')

print('Unique Values of Each Features !!')

for col in train.columns:

    print('Unique Values of the Field ', col, ' : ',train[col].nunique())
#Distribution by Vendor ID

dist = pd.DataFrame(train['vendor_id'].value_counts()).reset_index()

dist.columns = ['vendor_id', 'count']

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('dark')



plt.bar(dist['vendor_id'], dist['count'])

plt.show()
#Range of pickup and dropoff time

import datetime

train['pickup_datetime'] = train['pickup_datetime'].map(lambda x : datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

train['dropoff_datetime'] = train['dropoff_datetime'].map(lambda x : datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))



print('Start Time for Pick Up : ', min(train['pickup_datetime']))

print('End Time for Pick Up : '  , max(train['pickup_datetime']))



print('Start Time for Drop Off : ', min(train['dropoff_datetime']))

print('End Time for Drop Off  : ', max(train['dropoff_datetime']))
#Plot Distribution

train['pickup_month'] = train['pickup_datetime'].dt.month

train['pickup_dom'] = train['pickup_datetime'].dt.day

train['pickup_dow'] = train['pickup_datetime'].dt.weekday



train['dropoff_month'] = train['dropoff_datetime'].dt.month

train['dropoff_dom'] = train['dropoff_datetime'].dt.day

train['dropoff_dow'] = train['dropoff_datetime'].dt.weekday



#Monthly Distribution

dist = pd.DataFrame(train['pickup_month'].value_counts().reset_index())

dist.columns = ['month', 'count']

plt.bar(dist['month'], dist['count'])

plt.show()



dist = pd.DataFrame(train['dropoff_month'].value_counts().reset_index())

dist.columns = ['month', 'count']

plt.bar(dist['month'], dist['count'])

plt.show()



#Day of Month Distribution

dist = pd.DataFrame(train['pickup_dom'].value_counts().reset_index())

dist.columns = ['dom', 'count']

plt.bar(range(dist.shape[0]), dist['count'])

plt.xticks(range(dist.shape[0]), dist['dom'])

plt.xticks(fontsize = 10, rotation = 90)

plt.show()



dist = pd.DataFrame(train['dropoff_dom'].value_counts().reset_index())

dist.columns = ['dom', 'count']

plt.bar(range(dist.shape[0]), dist['count'])

plt.xticks(range(dist.shape[0]), dist['dom'])

plt.xticks(fontsize = 10, rotation = 90)

plt.show()



#Day of week Distribution

dist = pd.DataFrame(train['pickup_dow'].value_counts().reset_index())

dist.columns = ['dow', 'count']

plt.bar(range(dist.shape[0]), dist['count'])

plt.xticks(range(dist.shape[0]), dist['dow'])

plt.xticks(fontsize = 10, rotation = 90)

plt.show()



dist = pd.DataFrame(train['dropoff_dow'].value_counts().reset_index())

dist.columns = ['dow', 'count']

plt.bar(range(dist.shape[0]), dist['count'])

plt.xticks(range(dist.shape[0]), dist['dow'])

plt.xticks(fontsize = 10, rotation = 90)

plt.show()
#Distirbution of Passenger Count

dist = pd.DataFrame(train['passenger_count'].value_counts().reset_index())

dist.columns = ['passenger_count', 'count']

plt.bar(range(dist.shape[0]), dist['count'])

plt.xticks(range(dist.shape[0]), dist['passenger_count'])

plt.xticks(fontsize = 10, rotation = 90)

plt.show()
#Analyze Extreme Cases(Passenger Count 0, 7, 8, 9)

trainSub = train.query('passenger_count==0')

print(trainSub.head(5))

print('Unique Values of Each Features !!')

for col in train.columns:

    print('Unique Values of the Field ', col, ' : ',trainSub[col].nunique())
#Range of Pickup and Dropoff location

print('Range of PickUp Longitude %f and %f' %(min(train['pickup_longitude']),max(train['pickup_longitude'])))

print('Range of Dropoff Longitude %f and %f' %(min(train['dropoff_longitude']),max(train['dropoff_longitude'])))



print('Range of PickUp Latitude %f and %f' %(min(train['pickup_latitude']),max(train['pickup_latitude'])))

print('Range of Dropoff Latitude %f and %f' %(min(train['dropoff_latitude']),max(train['dropoff_latitude'])))
#Distribution of store and forward flag

print(train['store_and_fwd_flag'].value_counts())

dist = pd.DataFrame(train['store_and_fwd_flag'].value_counts().reset_index())

dist.columns = ['store_and_fwd_flag', 'count']

plt.bar(range(dist.shape[0]), dist['count'])

plt.xticks(range(dist.shape[0]), dist['store_and_fwd_flag'])

plt.xticks(fontsize = 10, rotation = 90)

plt.show()
#Convert seconds to minutes

train['trip_duration_minutes'] = train['trip_duration'].map(lambda x : x/60.)



#Range of trip duration

print('Range of Trip Duration : Min %f Max %f' %(min(train['trip_duration_minutes']), max(train['trip_duration_minutes'])))



#Distribution of trip duration

print(train['trip_duration_minutes'].value_counts())