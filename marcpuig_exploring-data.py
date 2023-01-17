%matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

import pandas as pd
import numpy as np
import datetime
### Load the CSV files specifying which is the datetime column and the parse function
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

### Convert the timestamp column to a datetime value 
df_train["time"] = pd.to_datetime(df_train["time"])
df_test["time"] = pd.to_datetime(df_train["time"])
### Let's see some context info about the dataset

print("Size of training data: %d" % df_train.shape[0])
print("Size of testing data: %d" % df_test.shape[0])
print("Number of different places: %d" % len(df_train['place_id'].unique()))
print("\nDataset Description:")
print(df_train.describe())
print("\nInfo about the time period:\n" + str(df_train['time'].describe()))
### I add some extra columns to expand the model with time related fields.
### This will be useful to plot some charts to understand the data.
df_train['hour'] = df_train['time'].dt.hour
df_train['weekday'] = df_train['time'].dt.dayofweek
### Let's visualize which days of the week are the most "active"
n, bins, patches = plt.hist(df_train['weekday'], 50)
plt.title('Day of the week')
plt.xlabel('Day of week')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
### We realize that there are some days with more data than the others
### Maybe this is because there's 10 days of data. 
### To discover more info about the dataset, we need to have uniform data. 
### We drop the days 8,9 and 10

df_train = df_train[df_train['time']<'1970-01-08 00:00:00']

print("Size of training data: %d" % df_train.shape[0])
print("Number of different places: %d" % len(df_train['place_id'].unique()))
print("\nDataset Description:")
print(df_train.describe())
print("\nInfo about the time period:\n" + str(df_train['time'].describe()))
### Visually check whether the weekday data is more uniform
n, bins, patches = plt.hist(df_train['weekday'], 50)
plt.title('Day of the week')
plt.xlabel('Day of week')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
### And the same exploration by weekday
### based on localization info

fig = plt.figure(figsize=(15,15))

for day in range(7):
    sp = fig.add_subplot(3,3,day+1)

    x = df_train[df_train['weekday'] == day]['x']
    y = df_train[df_train['weekday'] == day]['y']

    sp.hist2d(x, y, bins=20, norm=LogNorm())
    sp.set_title('x and y location histogram day %d' % day)
    sp.set_xlabel('x')
    sp.set_ylabel('y')

plt.show()
### To finalize this initial exploration, we plot some locations to find some patterns.

grouped = df_train['place_id'].value_counts().reset_index()
ids = grouped['index'][:10]

colors = cm.rainbow(np.linspace(0, 1, len(ids)))

plt.figure(figsize=(10,10))
for id, c in zip(ids, colors):
    x = df_train[df_train['place_id'] == id]['x']
    y = df_train[df_train['place_id'] == id]['y']
    plt.scatter(x, y, color=c)
    
plt.grid(True)
plt.xlim(-0.1,10)
plt.ylim(-0.1,10)
plt.show()
### Interesting!