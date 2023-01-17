import datetime

import os



import matplotlib.pyplot as plt

import pandas as pd

from dateutil import tz

from matplotlib.dates import DateFormatter

from pandas.plotting import register_matplotlib_converters

from scipy.stats import norm

register_matplotlib_converters()



%matplotlib inline
# data file location(s)

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# load data

wireless_df = pd.read_csv('/kaggle/input/ping-data/ping_data_rp829c7e0e.csv')

ethernet_df = pd.read_csv('/kaggle/input/ping-data/ping_data_rp829c7e0e.csv')
# convert timestamps to datetime

wireless_df['timestamp'] = pd.to_datetime(wireless_df['timestamp'], unit='s')

wireless_df['timestamp'] = wireless_df.timestamp.dt.tz_localize('UTC').dt.tz_convert('US/Eastern')



ethernet_df['timestamp'] = pd.to_datetime(ethernet_df['timestamp'], unit='s')

ethernet_df['timestamp'] = ethernet_df.timestamp.dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
# timespan of time series data

timespan = wireless_df['timestamp'][-1:] - wireless_df['timestamp'][0]

print('Start Date:', wireless_df['timestamp'].iloc[0])

print('  End Date:', wireless_df['timestamp'].iloc[-1])

print('  Timespan:', timespan.iloc[-1])
# expected/ideal number of data points within timespan

sensor_frequency = 10

expected_data_points = timespan / datetime.timedelta(seconds=sensor_frequency)

int(expected_data_points.iloc[-1])
# number of missing data points

missing_data_points = len(wireless_df.index) - int(expected_data_points)

int(missing_data_points)
# minutes worth of missing data

missing_time = (missing_data_points * sensor_frequency) / 60 # missing minutes

int(missing_time)
wireless_df.describe()
wireless_df.head()
# clip loss of network (9999.9999) and timeouts (2000.0)

wireless_df = wireless_df[wireless_df['local_avg'] < 2000]

wireless_df = wireless_df[wireless_df['remote_avg'] < 2000]



ethernet_df = ethernet_df[ethernet_df['local_avg'] < 2000]

ethernet_df = ethernet_df[ethernet_df['remote_avg'] < 2000]
wireless_df.describe()
# clip outliers (anomalies)

qt = wireless_df['local_avg'].quantile(0.98)

wireless_df = wireless_df[wireless_df['local_avg'] <= qt]



qh = wireless_df['remote_avg'].quantile(0.98)

wireless_df = wireless_df[wireless_df['remote_avg'] <= qh]
wireless_df.describe()
x = wireless_df.loc[wireless_df['local_avg'] <= 2.0, ['local_avg']] # data distribution

mu = x.mean() # mean of distribution

sigma = x.std() # standard deviation of distribution



print('data: local_avg')

print('mu (\u03BC): %.2f' % mu)

print('sigma (\u03C3): %.2f' % sigma)
# plot network data

wireless_df = wireless_df[(wireless_df['timestamp'] >= '2019-06-02') & (wireless_df['timestamp'] < '2019-06-06')]



_, ax = plt.subplots(1, 1, figsize=(15, 12))

ax.plot(wireless_df['timestamp'], wireless_df['remote_avg'], linestyle=' ', marker='.', alpha=0.5, label='remote')

ax.plot(wireless_df['timestamp'], wireless_df['local_avg'], linestyle=' ', marker='.', alpha=0.5, label='local')

ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H:%M', tz=tz.gettz('US/Eastern')))

ax.legend()

ax.set_title('Wireless Ping Response Times', fontsize=16)

ax.set_xlabel('date/time', fontsize=12)

ax.set_ylabel('response [ms]', fontsize=12)

ax.grid(color='silver', linestyle='solid', linewidth=1, alpha=0.5)



plt.show()
# plot network data

ethernet_df = ethernet_df[(ethernet_df['timestamp'] >= '2019-06-02') & (ethernet_df['timestamp'] < '2019-06-06')]



_, ax = plt.subplots(1, 1, figsize=(15, 12))

ax.plot(ethernet_df['timestamp'], ethernet_df['remote_avg'], linestyle=' ', marker='.', alpha=0.5, label='remote')

ax.plot(ethernet_df['timestamp'], ethernet_df['local_avg'], linestyle=' ', marker='.', alpha=0.5, label='local')

ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H:%M', tz=tz.gettz('US/Eastern')))

ax.legend()

ax.set_title('Ethernet Ping Response Times', fontsize=16)

ax.set_xlabel('date/time', fontsize=12)

ax.set_ylabel('response [ms]', fontsize=12)

ax.grid(color='silver', linestyle='solid', linewidth=1, alpha=0.5)



plt.show()
# calculate time delta between consecutive rows

wireless_df['time_delta'] = wireless_df['timestamp'].diff()

ethernet_df['time_delta'] = ethernet_df['timestamp'].diff()
wireless_df.sort_values(by='time_delta', ascending=False).head()
# identify time series gaps > 60 seconds

min_threshold = 60

columns = ['timestamp', 'time_delta']

gaps_wireless_df = wireless_df.loc[wireless_df.time_delta > datetime.timedelta(seconds=min_threshold), columns]
gaps_wireless_df.sort_values(by='time_delta', ascending=False)
gaps_wireless_df.describe()
# plot time series gaps

_, ax = plt.subplots(1, 1, figsize=(15, 12))

ax.plot(gaps_wireless_df['timestamp'], gaps_wireless_df['time_delta'].astype('timedelta64[m]'), 

        linestyle=' ', marker='v', markersize=14, markerfacecolor='r', markeredgecolor='k', 

        alpha=0.75, label='gaps')



ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H:%M', tz=tz.gettz('US/Eastern')))

ax.legend()

ax.set_title('Time Gaps > 1 minute', fontsize=16)

ax.set_xlabel('date/time', fontsize=12)

ax.set_ylabel('gaps [min]', fontsize=12)

ax.grid(color='silver', linestyle='solid', linewidth=1, alpha=0.5)

plt.show()