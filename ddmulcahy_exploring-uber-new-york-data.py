import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
april_uber = pd.read_csv('../input/uber-raw-data-apr14.csv')

april_uber.info()
april_uber['Date/Time'] = pd.to_datetime(april_uber['Date/Time'], format="%m/%d/%Y %H:%M:%S")
april_uber['Dayofweek'] = april_uber['Date/Time'].dt.dayofweek

april_uber['Daynumber'] = april_uber['Date/Time'].dt.day

april_uber['Hour'] = april_uber['Date/Time'].dt.hour
print(april_uber['Dayofweek'].unique())

print(april_uber['Daynumber'].unique())

print(april_uber['Hour'].unique())
april_uber['Hour'].hist(bins=24,color='k', alpha=0.5)

plt.xlim(0,23)

#plt.title('')

plt.ylabel('Total Journeys')

plt.xlabel('Time in Hours');

plt.axvline(8, color='r', linestyle='solid')

plt.axvline(17, color='r', linestyle='solid')
april_uber['Daynumber'].hist(bins=30,color='k', alpha=0.5)

plt.xlim(1,30)

plt.title('Journeys for April 2014')

plt.ylabel('Total Journeys')

plt.xlabel('Date in April');
april_uber[april_uber['Daynumber']==12]['Hour'].hist(bins=24,color='k', alpha=0.5)

plt.xlim(0,24)

plt.title('Journeys for 12th of April 2014')

plt.ylabel('Total Journeys')

plt.xlabel('Time in Hours');

plt.axvline(8, color='r', linestyle='solid')

plt.axvline(17, color='r', linestyle='solid')
test_30 = april_uber[april_uber['Daynumber']== 30]

test_23 = april_uber[april_uber['Daynumber']== 23]

print(len(test_30))

print(len(test_23))
test_30['Hour'].hist(bins=24,color='k', alpha=0.5)

test_23['Hour'].hist(bins=24,color='b', alpha=0.5)

plt.xlim(0,23)

#plt.title('')

plt.ylabel('Total Journeys')

plt.xlabel('Time in Hours');

plt.axvline(8, color='r', linestyle='solid')

plt.axvline(17, color='r', linestyle='solid')
#should write a simple function for this but maybe later when I have finished the full analysis. 

may_uber = pd.read_csv('../input/uber-raw-data-may14.csv')

may_uber['Date/Time'] = pd.to_datetime(may_uber['Date/Time'], format="%m/%d/%Y %H:%M:%S")

may_uber['Dayofweek'] = may_uber['Date/Time'].dt.dayofweek

may_uber['Daynumber'] = may_uber['Date/Time'].dt.day

may_uber['Hour'] = may_uber['Date/Time'].dt.hour

jun_uber = pd.read_csv('../input/uber-raw-data-jun14.csv')

jun_uber['Date/Time'] = pd.to_datetime(jun_uber['Date/Time'], format="%m/%d/%Y %H:%M:%S")

jun_uber['Dayofweek'] = jun_uber['Date/Time'].dt.dayofweek

jun_uber['Daynumber'] = jun_uber['Date/Time'].dt.day

jun_uber['Hour'] = jun_uber['Date/Time'].dt.hour

jul_uber = pd.read_csv('../input/uber-raw-data-jul14.csv')

jul_uber['Date/Time'] = pd.to_datetime(jul_uber['Date/Time'], format="%m/%d/%Y %H:%M:%S")

jul_uber['Dayofweek'] = jul_uber['Date/Time'].dt.dayofweek

jul_uber['Daynumber'] = jul_uber['Date/Time'].dt.day

jul_uber['Hour'] = jul_uber['Date/Time'].dt.hour

aug_uber = pd.read_csv('../input/uber-raw-data-aug14.csv')

aug_uber['Date/Time'] = pd.to_datetime(aug_uber['Date/Time'], format="%m/%d/%Y %H:%M:%S")

aug_uber['Dayofweek'] = aug_uber['Date/Time'].dt.dayofweek

aug_uber['Daynumber'] = aug_uber['Date/Time'].dt.day

aug_uber['Hour'] = aug_uber['Date/Time'].dt.hour

sep_uber = pd.read_csv('../input/uber-raw-data-sep14.csv')

sep_uber['Date/Time'] = pd.to_datetime(sep_uber['Date/Time'], format="%m/%d/%Y %H:%M:%S")

sep_uber['Dayofweek'] = sep_uber['Date/Time'].dt.dayofweek

sep_uber['Daynumber'] = sep_uber['Date/Time'].dt.day

sep_uber['Hour'] = sep_uber['Date/Time'].dt.hour
full_uber = pd.concat([april_uber,may_uber,jun_uber,jul_uber,aug_uber,sep_uber])
full_uber['Month'] = full_uber['Date/Time'].dt.month
full_uber['Hour'].hist(bins=24,color='k', alpha=0.5)

plt.xlim(0,23)

plt.ylabel('Total Journeys')

plt.xlabel('Time in Hours');

plt.axvline(8, color='r', linestyle='solid')

plt.axvline(17, color='r', linestyle='solid')
full_uber['Month'].hist(bins=6,color='k', alpha=0.5)

plt.xlim(4,9)

#plt.title('')

plt.ylabel('Total Journeys')

plt.xlabel('Month');
((len(full_uber[full_uber['Month']==9])-len(full_uber[full_uber['Month']==4]))/len(full_uber[full_uber['Month']==4])) * 100
skyline = pd.read_csv('../input/other-Skyline_B00111.csv')

skyline['Date'] = pd.to_datetime(skyline['Date'], format="%m/%d/%Y")

skyline['Month'] = skyline['Date'].dt.month

full_uber['Month'].hist(bins=6,color='k', alpha=0.5)

skyline['Month'].hist(bins=3,color='k', alpha=0.5)

plt.xlim(4,9)

#plt.title('')

plt.ylabel('Total Journeys')

plt.xlabel('Time in Hours');
print(len(skyline[skyline['Month']==7]))

print(len(skyline[skyline['Month']==8]))

print(len(skyline[skyline['Month']==9]))