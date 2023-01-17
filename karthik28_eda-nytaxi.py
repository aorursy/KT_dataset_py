# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Read the train and test data file

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print("Total number of samples in train file : ", train.shape[0])

print("Total number of samples in test file : ", test.shape[0])
# Let's look at the train data first

print("A view of the train dataframe")

print(train.head())

print("\nColumns in train dataset : ", train.columns)

print("\n")

print("Overall description of the train dataset : ")

print(train.info())
# Okay  we have an overview of the train dataset. Before exploring further, let's see if the id column has some overlap with the test ids or not

train_id = set(train['id'].values)

test_id = set(test['id'].values)

print("Number of unique id in train dataset : ", len(train_id))

print("Number of unique id in test dataset : ", len(test_id))

common_ids = train_id.intersection(test_id)

print("Number of common id in the train and test datasets : ", len(common_ids))
# Let's have a look at the traget variable(trip duration) first

target = train['trip_duration']

print("Longest trip duration {} or {} minutes: " .format(np.max(target.values), np.max(target.values)//60))

print("Smallest trip duration {} or {} minutes: ".format(np.min(target.values),np.min(target.values)//60))

print("Average trip duration : {} or {} minutes".format(np.mean(target.values), np.mean(target.values)//60))
#Visualization is always better 

f = plt.figure(figsize=(8,6))

plt.scatter(range(len(target)), np.sort(target.values), alpha=0.5)

plt.xlabel('Index')

plt.ylabel('Trip duration in seconds')

plt.show()
# Moving on to the vendor_id column

unique_vendors = set(train['vendor_id'].values)

print("Number of unique vendors : ", len(unique_vendors))

print("How popular is the vendor? ")

vendor_popularity = train['vendor_id'].value_counts()



f = plt.figure(figsize=(10,5))

sns.barplot(vendor_popularity.index, vendor_popularity.values, alpha=0.7)

plt.xlabel('Vendor', fontsize=14)

plt.ylabel('Trips offered', fontsize=14)

plt.show()
# Moving to passengers count column

pass_count = train['passenger_count']

print("Maximum number of passengers on a trip : ", np.max(pass_count.values))

print("Minimum number of passengers on a trip : ", np.min(pass_count.values))

print("Average number of passengers on a trip : ", np.mean(pass_count.values))



f = plt.figure(figsize=(10,5))

pass_count = train['passenger_count'].value_counts()

sns.barplot(pass_count.index, pass_count.values, alpha=0.7)

plt.xlabel('Number of passengers on a trip', fontsize=14)

plt.ylabel('Count', fontsize=14)

plt.show()
# Let's move to the store_and_fwd_flag column

flags = train['store_and_fwd_flag'].value_counts()



f = plt.figure(figsize=(10,5))

sns.barplot(flags.index, flags.values, alpha=0.7)

plt.xlabel('Flags', fontsize=14)

plt.ylabel('Count', fontsize=14)

plt.show()
train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])

train['dropoff_datetime'] = pd.to_datetime(train['dropoff_datetime'])



train['pickup_day'] = train['pickup_datetime'].dt.day

train['pickup_month'] = train['pickup_datetime'].dt.month

train['pickup_weekday'] = train['pickup_datetime'].dt.weekday

train['pickup_hour'] = train['pickup_datetime'].dt.hour



train['drop_day'] = train['dropoff_datetime'].dt.day

train['drop_month'] = train['dropoff_datetime'].dt.month

train['drop_weekday'] = train['dropoff_datetime'].dt.weekday

train['drop_hour'] = train['dropoff_datetime'].dt.hour
# Do the number of pickups looks same for the whole month? Let's find out

f = plt.figure(figsize=(15,5))

sns.countplot(x='pickup_day', data=train)

plt.xlabel('Day of month', fontsize=14)

plt.ylabel('Pickup count', fontsize=14)

plt.show()
f = plt.figure(figsize=(15,5))

sns.countplot(x='pickup_month', data=train)

plt.xlabel('Month', fontsize=14)

plt.ylabel('Pickup count', fontsize=14)

plt.show()
f = plt.figure(figsize=(15,5))

days = [i for i in range(7)]

sns.countplot(x='pickup_weekday', data=train)

plt.xlabel('Day of the week', fontsize=14)

plt.ylabel('Pickup count', fontsize=14)

plt.xticks(days, ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'))

plt.show()
f = plt.figure(figsize=(15,5))

sns.countplot(x='pickup_hour', data=train)

plt.xlabel('Hour', fontsize=14)

plt.ylabel('Pickup count', fontsize=14)

plt.show()
f = plt.figure(figsize=(10,8))

days = [i for i in range(7)]

sns.countplot(x='pickup_weekday', data=train, hue='pickup_hour', alpha=0.8)

plt.xlabel('Day of the week', fontsize=14)

plt.ylabel('Pickup count', fontsize=14)

plt.xticks(days, ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'))

plt.legend(loc=(1.04,0))

plt.show()