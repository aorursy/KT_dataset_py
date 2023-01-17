import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
# load the data
elec_data = pd.read_csv('../input/TimeSeries_TotalSolarGen_and_Load_IT_2015.csv')
# ensure the data is sorted by timestamp
elec_data = elec_data.sort_values(by=['utc_timestamp'])
# Show an interesting slice of the data
elec_data[75:90]
# check data types
elec_data.dtypes
# add a timestamp column
elec_data['utc_timestamp'] = pd.to_datetime(elec_data['utc_timestamp'])
# examples for formatting the conversion can be found at http://strftime.org
# check data types again
elec_data.dtypes
elec_data.describe()
fig = plt.figure(figsize=(12, 5))
sns.distplot(elec_data['IT_solar_generation'].loc[elec_data['IT_solar_generation']<100], bins=50)
# only keep data above a threshold
elec_data = elec_data.loc[elec_data['IT_solar_generation'] > 2]
# calculate the difference of timestamps between rows
elec_data['time_diff'] = elec_data['utc_timestamp'].diff()
# calculate an integer of seconds for the difference
elec_data['time_diff_int'] = elec_data.time_diff / np.timedelta64(1, 's')
# calculate a group ID for all intervals that fall contiguously in time
elec_data["group_id"] = elec_data['time_diff_int'].diff().ne(0).cumsum()
elec_data.head(20)
# add on a bunch of broken out columns for the individual time elements
elec_data["year"] = elec_data['utc_timestamp'].dt.year
elec_data["month"] = elec_data['utc_timestamp'].dt.month
elec_data["hour"] = elec_data['utc_timestamp'].dt.hour
elec_data["dayofweek"] = elec_data['utc_timestamp'].dt.dayofweek
elec_data.head(20)
# aggregate the data and save as a new data frame
exec_start_time = time.time()
elec_data_grouped = elec_data.groupby(['group_id','dayofweek','month'], 
                                      as_index=False).agg({'IT_solar_generation':sum,
                                                           'utc_timestamp':"count"})

# another slightly slower option instead of group by
'''
elec_data_grouped = elec_data.pivot_table(index=['group_id','dayofweek','month'],
                                          values=['IT_solar_generation', 'utc_timestamp'],
                                          aggfunc={'IT_solar_generation':sum,
                                                   'utc_timestamp':"count"})
elec_data_grouped = elec_data_grouped.rename_axis(None, axis=1).reset_index()
'''

exec_end_time = time.time()
print("Elapsed time was %g seconds" % (exec_end_time - exec_start_time))

elec_data_grouped.rename(columns={"utc_timestamp": "contiguous_interval_count",
                                  "IT_solar_generation": "group_solar_sum"}, 
                         inplace=True)
elec_data_grouped.head(20)
# plot the data
fig = plt.figure(figsize=(12, 5))
plt.title('Groups of contiguous hours of solar production')
sns.countplot(data = elec_data_grouped, x='contiguous_interval_count')
plot_data = elec_data_grouped.loc[elec_data_grouped['contiguous_interval_count'].between(9, 17, inclusive=True)]
fig, ax = plt.subplots(ncols=1, figsize=(14,6))
s = sns.boxplot(ax = ax, 
                x="month", 
                y="group_solar_sum", 
                hue="contiguous_interval_count",
                data=plot_data, 
                palette="PRGn",
                showfliers=False)
plt.show();