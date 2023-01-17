import random

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Supress some errors

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()



plt.style.use('bmh')

%matplotlib inline
train = pd.read_csv("/kaggle/input/ashrae-energy-prediction/train.csv")

train.head()
print(f"Number of data points in training set {len(train)}")
pd.Series(train['building_id'], dtype="category").describe()
print(f"Number of unique building_ids in training set {pd.Series(train['building_id'], dtype='category').describe()['unique']}")
plt.figure(figsize=(10,6))

plt.hist(list(train['building_id'].value_counts()), bins=14)

plt.title("Number of data points for each building_id")

plt.xlabel("Number of data points ")

plt.ylabel("Number of building with that many data points")

plt.show()
train['building_id'].value_counts().describe()
train['timestamp'] = pd.to_datetime(train['timestamp'])

train['timestamp'].describe()
print(f"Number of unique timestamps is {train['timestamp'].describe()['unique']}")

print(f"last  timestamp is {train['timestamp'].describe()['last']}")

print(f"first timestamp is {train['timestamp'].describe()['first']}")
tcounts = train['timestamp'].value_counts()

plt.figure(figsize=(10,6))

plt.plot(tcounts.index, tcounts, '*')

plt.title("data distribution wrt time")

plt.xlabel("time stamp")

plt.ylabel("data points")

plt.show()
train['meter'].value_counts()
plt.figure(figsize=(10,6))

plt.bar(["electricity","chilledwater", "steam", "hotwater"], train['meter'].value_counts())

plt.xlabel("Meter ids")

plt.ylabel("counts")

plt.show()
print("Number of data points for each unique meter")

print((train["meter"] + train["building_id"]*10).value_counts().describe())
plt.figure(figsize=(10,6))

plt.hist((train["meter"] + train["building_id"]*10).value_counts(), bins=50)

plt.title("Number of data points for each unique meter")

plt.xlabel("Number of data points ")

plt.ylabel("Number of unique meters with that many data points")

plt.show()
train['meter_reading'].describe()
nonzero = train['meter_reading'][train['meter_reading'] != 0.0]
print(f"total number of readings          {len(train['meter_reading'])}")

print(f"total number of non-zero readings {len(nonzero)}")
# ploting non zero values less than 1000 as very high values screw the plot

# len(nonzero[nonzero <= 1000] is 16723689

plt.figure(figsize=(10,10))

plt.hist(nonzero[nonzero <= 1000], bins=100)

plt.title("non-zero meter readings  < 1000")

plt.show()
fig, axes = plt.subplots(2, 2, figsize=(15,10))



a = train['meter_reading'][train['meter'] == 0]

nonzero = a[ a != 0.0]

print(f"total number of readings for meter type electricity          {len(a)}")

print(f"total number of non-zero readings for meter type electricity {len(nonzero)}")

axes[0,0].hist(nonzero[nonzero <= 1000], bins=40)

axes[0,0].set_title("meter readings for meter type electricity");



a = train['meter_reading'][train['meter'] == 1]

nonzero = a[ a != 0.0]

print(f"total number of readings for meter type chilledwater          {len(a)}")

print(f"total number of non-zero readings for meter type chilledwater {len(nonzero)}")

axes[0,1].hist(nonzero[nonzero <= 1000], bins=40)

axes[0,1].set_title("meter readings for meter type chilledwater")



a = train['meter_reading'][train['meter'] == 2]

nonzero = a[ a != 0.0]

print(f"total number of readings for meter type steam                 {len(a)}")

print(f"total number of non-zero readings for meter type steam        {len(nonzero)}")

axes[1,0].hist(nonzero[nonzero <= 1000], bins=40)

axes[1,0].set_title("meter readings for meter type steam")



a = train['meter_reading'][train['meter'] == 3]

nonzero = a[ a != 0.0]

print(f"total number of readings for meter type hotwater              {len(a)}")

print(f"total number of non-zero readings for meter type hotwater     {len(nonzero)}")

axes[1,1].hist(nonzero[nonzero <= 1000], bins=40)

axes[1,1].set_title("meter readings for meter type hotwater")





plt.tight_layout()

plt.show()
train['unique_meter'] = train["meter"] + train["building_id"]*10



k = 5

fig, axes = plt.subplots(5, 1, figsize=(5*k, 10), sharex=True)

fig.suptitle("Some randomly sampled Series of meter reading over entire time period")

l = list(train['unique_meter'].unique())

for i, j in enumerate(random.choices(l, k=k)):

    x = train[train['unique_meter']==j]

    axes[i].plot(x['timestamp'], x['meter_reading'], alpha=0.9)

plt.show()
k = 5

days = 15

fig, axes = plt.subplots(5, 1, figsize=(5*k, 10))

fig.suptitle(f"Some randomly sampled Series of meter reading over {days} days")

l = list(train['unique_meter'].unique())

for i, j in enumerate(random.choices(l, k=k)):

    x = train[train['unique_meter']==j]

    axes[i].plot(x['timestamp'][:24*days], x['meter_reading'][:24*days], alpha=0.9)

plt.show()
train['log_meter_reading'] = np.log1p(train["meter_reading"])

nonzero = train['log_meter_reading'][train['log_meter_reading'] != 0.0]

plt.figure(figsize=(10,10))

plt.hist(nonzero, bins=100)

plt.title("non-zero log1p meter readings")

plt.show()
fig, axes = plt.subplots(2, 2, figsize=(15,10))



a = train['log_meter_reading'][train['meter'] == 0]

nonzero = a[ a != 0.0]

axes[0,0].hist(nonzero, bins=40)

axes[0,0].set_title("log1p meter readings for meter type electricity");



a = train['log_meter_reading'][train['meter'] == 1]

nonzero = a[ a != 0.0]

axes[0,1].hist(nonzero, bins=40)

axes[0,1].set_title("log1p meter readings for meter type chilledwater")



a = train['log_meter_reading'][train['meter'] == 2]

nonzero = a[ a != 0.0]

axes[1,0].hist(nonzero, bins=40)

axes[1,0].set_title("log1p meter readings for meter type steam")



a = train['log_meter_reading'][train['meter'] == 3]

nonzero = a[ a != 0.0]

axes[1,1].hist(nonzero, bins=40)

axes[1,1].set_title("log1p meter readings for meter type hotwater")



plt.tight_layout()

plt.show()
k = 5

fig, axes = plt.subplots(5, 1, figsize=(5*k, 10), sharex=True)

fig.suptitle("Some randomly sampled Series of meter reading over entire time period")

l = list(train['unique_meter'].unique())

for i, j in enumerate(random.choices(l, k=k)):

    x = train[train['unique_meter']==j]

    axes[i].plot(x['timestamp'], x['log_meter_reading'], alpha=0.9)

plt.show()
k = 5

days = 15

fig, axes = plt.subplots(5, 1, figsize=(5*k, 10))

fig.suptitle(f"Some randomly sampled Series of meter reading over {days} days")

l = list(train['unique_meter'].unique())

for i, j in enumerate(random.choices(l, k=k)):

    x = train[train['unique_meter']==j]

    axes[i].plot(x['timestamp'][:24*days], x['log_meter_reading'][:24*days], alpha=0.9)

plt.show()
test = pd.read_csv("/kaggle/input/ashrae-energy-prediction/test.csv")

test.head()
print(f"Number of data points in testing set {len(test)}")
pd.Series(test['building_id'], dtype="category").describe()
if np.all(test['building_id'].value_counts().index.sort_values() == train['building_id'].value_counts().index.sort_values()):

    print("the building_id's in train and test match")
plt.figure(figsize=(10,6))

plt.hist(list(test['building_id'].value_counts()))

plt.title("Number of data points for each building_id")

plt.xlabel("Number of data points")

plt.ylabel("Number of building with that many data points")

plt.show()
test['building_id'].value_counts().value_counts()
timestamps = pd.to_datetime(test['timestamp'])

timestamps.describe()
print(f"Number of unique timestamps is {timestamps.describe()['unique']}")

print(f"last  timestamp is {timestamps.describe()['last']}")

print(f"first timestamp is {timestamps.describe()['first']}")
tcounts = timestamps.value_counts()

plt.figure(figsize=(10,6))

plt.plot(tcounts.index, tcounts, '*')

plt.title("data distribution wrt time")

plt.xlabel("time stamp")

plt.ylabel("data points")

plt.show()
test['meter'].value_counts() 
plt.figure(figsize=(10,6))

plt.bar(["electricity","chilledwater", "steam", "hotwater"], test['meter'].value_counts())

plt.xlabel("Meter ids")

plt.ylabel("counts")

plt.show()
if np.array(train["meter"] + train["building_id"]*10).sort() == np.array(test["meter"] + test["building_id"]*10).sort():

    print("the meters in test set are all present in training set")
print("Number of data points for each unique meter")

print((test["meter"] + test["building_id"]*10).value_counts().describe())
meta = pd.read_csv("/kaggle/input/ashrae-energy-prediction/building_metadata.csv")

meta.head()
nul = meta.isnull()

for i in nul.columns:

    print(f"column name:{i}\t len:{len(nul[i])} number of null values : {np.sum(nul[i])}")
if np.all(meta['building_id'].value_counts().index.sort_values() == train['building_id'].value_counts().index.sort_values()):

    print("the building_id's in train and meta_data match")
pd.Series(meta['primary_use'], dtype="category").value_counts()
plt.figure(figsize=(8,10))

plt.barh(meta['primary_use'].apply(str).value_counts().index, meta['primary_use'].value_counts())

plt.ylabel("primary use")

plt.xlabel("counts")

plt.show()
meta['square_feet'].describe()
plt.figure(figsize=(10,6))

plt.hist(meta['square_feet'], bins=20)

plt.ylabel("bin counts")

plt.xlabel("square feet")

plt.show()
year = meta['year_built']

year.describe()
plt.figure(figsize=(15,6))

plt.hist(meta['year_built'], bins=2017-1900-1, rwidth=0.8)

plt.ylabel("bin counts")

plt.xlabel("year")

plt.title("Year built")

plt.show()
meta['floor_count'].value_counts()
nonnull = meta['floor_count'][~ meta['floor_count'].isnull()]

plt.figure(figsize=(10,6))

plt.bar(nonnull.apply(str).value_counts().index, nonnull.value_counts())

plt.xlabel("Floor counts")

plt.ylabel("Instances")

plt.show()
wtrain = pd.read_csv("/kaggle/input/ashrae-energy-prediction/weather_train.csv")

wtrain['timestamp'] = pd.to_datetime(wtrain['timestamp'])

wtrain.head()
wtest = pd.read_csv("/kaggle/input/ashrae-energy-prediction/weather_test.csv")

wtest['timestamp'] = pd.to_datetime(wtest['timestamp'])

wtest.head()
print("weather train")

print("")

nul = wtrain.isnull()

for i in nul.columns:

    print(f"column name:{i: <{20}}len:{len(nul[i])} \tnumber of null values : {np.sum(nul[i])}")



print("")

print("-"*80)

print("weather test")

print("")

nul = wtest.isnull()

for i in nul.columns:

    print(f"column name:{i: <{20}}len:{len(nul[i])} \tnumber of null values : {np.sum(nul[i])}")

assert np.all(wtrain['site_id'].unique() == wtest['site_id'].unique())
wtrain.groupby("site_id")['timestamp'].describe()[[ 'count', 'unique']].head()
wtest.groupby("site_id")['timestamp'].describe()[[ 'count', 'unique']].head()
plt.figure(figsize=(9,4))

for i in range(16):

    site = wtrain[wtrain['site_id']== i]

    plt.plot(site['timestamp'], site['air_temperature'])

plt.xlabel("Time")

plt.ylabel("Temperature")

plt.title("Temperature at different sites in train data")

plt.show()
plt.figure(figsize=(20,6))

for i in range(16):

    site = wtest[wtest['site_id']== i]

    plt.plot(site['timestamp'], site['air_temperature'])

plt.xlabel("Time")

plt.ylabel("Temperature")

plt.title("Temperature at different sites in test data")

plt.show()
plt.figure(figsize=(9,4))

for i in range(16):

    site = wtrain[wtrain['site_id']== i]

    plt.plot(site['timestamp'], site['dew_temperature'])

plt.xlabel("Time")

plt.ylabel("Dew Temperature")

plt.title("Dew Temperature at different sites in train data")

plt.show()
plt.figure(figsize=(20,6))

for i in range(16):

    site = wtest[wtest['site_id']== i]

    plt.plot(site['timestamp'], site['dew_temperature'])

plt.xlabel("Time")

plt.ylabel("Dew Temperature")

plt.title("Dew Temperature at different sites in test data")

plt.show()
plt.figure(figsize=(9,4))

for i in range(16):

    site = wtrain[wtrain['site_id']== i]

    plt.plot(site['timestamp'], site['sea_level_pressure'], alpha=0.8)

plt.xlabel("Time")

plt.ylabel("sea_level_pressure")

plt.title("sea_level_pressure at different sites in train data")

plt.show()
plt.figure(figsize=(20,6))

for i in range(16):

    site = wtest[wtest['site_id']== i]

    plt.plot(site['timestamp'], site['sea_level_pressure'], alpha=0.8)

plt.xlabel("Time")

plt.ylabel("sea_level_pressure")

plt.title("sea_level_pressure at different sites in test data")

plt.show()
plt.figure(figsize=(9,4))

for i in range(16):

    site = wtrain[wtrain['site_id']== i]

    plt.plot(site['timestamp'], site['wind_speed'], alpha=0.8)

plt.xlabel("Time")

plt.ylabel("wind_speed")

plt.title("wind_speed at different sites in train data")

plt.show()
plt.figure(figsize=(20,6))

for i in range(16):

    site = wtest[wtest['site_id']== i]

    plt.plot(site['timestamp'], site['wind_speed'], alpha=0.8)

plt.xlabel("Time")

plt.ylabel("wind_speed")

plt.title("wind_speed at different sites in test data")

plt.show()
sub = pd.read_csv("/kaggle/input/ashrae-energy-prediction/sample_submission.csv")

sub.head()
if len(sub['row_id']) == len(test['row_id']):

    print("The lengths of submission.csv and test.csv match")