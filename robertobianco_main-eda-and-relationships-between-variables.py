import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

#from time import time

import datetime as dt
data_folder = '/kaggle/input/ashrae-energy-prediction/'

train_df = pd.read_csv(data_folder + 'train.csv')
train_df['log_meter_reading'] = np.log(1+train_df['meter_reading'])

# convert timestamp into datetime object

train_df['timestamp'] = pd.to_datetime(train_df['timestamp'],format='%Y-%m-%d %H:%M:%S')

train_df.head()
train_df.tail()
train_df.groupby('building_id')['meter'].nunique().hist(bins= np.arange(0.5,5.5,1));

plt.xticks(np.arange(1,5));

plt.ylabel('# of buildings')

plt.xlabel('# of different meter values per building');
building_meter_datapoints = train_df.groupby(['building_id','meter'])['meter_reading'].count().to_frame()

building_meter_datapoints.shape
building_meter_datapoints.head()
building_meter_datapoints.meter_reading.max()
max_data_points = 366*24

sum(building_meter_datapoints.meter_reading==max_data_points)
bins = np.concatenate( ( np.arange(0,9000,1000), np.array([max_data_points-.5]) ) )

max_bin = np.array([max_data_points-.5, max_data_points+100])



fig, ax = plt.subplots(figsize=(8,6))

ax.hist(building_meter_datapoints.meter_reading, bins=bins,histtype='bar')

ax.hist(building_meter_datapoints.meter_reading, bins=max_bin)

plt.grid();

plt.yscale('log');

plt.xlabel('# of data points');

plt.ylabel('building-meter occurrences');

# The red bar represents cases with complete dataset

# NOTE: the plot is in log-scale
building_meter_datapoints.loc[building_meter_datapoints.idxmin()]
b_403_0 = train_df[(train_df['building_id']==403) & (train_df['meter']==0)]

b_403_0.plot(x='timestamp',y='meter_reading', figsize=(8,6));

plt.grid();
# Read building metadata

building_df = pd.read_csv(data_folder + 'building_metadata.csv')

building_df['log_square_feet'] = np.log(building_df['square_feet'])
plt.hist(np.log10(1+train_df['meter_reading']))

plt.yscale('log');

plt.grid();

plt.xlabel('log10[meter_reading]')

plt.ylabel('datapoints');
plt.hist(np.log10(1+building_df['square_feet']))

plt.yscale('log');

plt.grid();

plt.xlabel('log10[square_feet]')

plt.ylabel('# of buildings');
trainbuilding_df = train_df.join(building_df, on='building_id', rsuffix = 'r')



trainbuilding_df[['meter_reading','square_feet']].corr()
trainbuilding_df[['log_meter_reading','log_square_feet']].corr()
trainbuilding_edu0_df = trainbuilding_df[(trainbuilding_df['primary_use']=='Education') & (trainbuilding_df['meter']==0)]

trainbuilding_edu0_df.head()
trainbuilding_edu0_df[['log_meter_reading','log_square_feet']].corr()