#Call required libraries

import numpy as np            # Data manipulation

from numpy import nan as NA

import pandas as pd           # Dataframe manipulatio 

from pandas import DataFrame

import matplotlib.pyplot as plt

import os

import conda

import seaborn as sns

from sklearn.cluster import KMeans #For clustering
# Load data

taxi_df = pd.read_csv('../input/taxi_final.csv', delimiter = ',', low_memory = True )
# Column names

taxi_df.columns
# Basic information about dataset

taxi_df.info()
# Drop useless columns

taxi_df.drop(['DateCreated', 'StartDateTime.1', 'Unnamed: 31'], axis=1, inplace=True)
# Remove nulls and wrong values

taxi_df = taxi_df[taxi_df['FareAmount'] > 0]

taxi_df = taxi_df[taxi_df['GratuityAmount'] >= 0]

taxi_df = taxi_df[taxi_df['SurchargeAmount'] >= 0]

taxi_df = taxi_df[taxi_df['ExtraFareAmount'] >= 0]

taxi_df = taxi_df[taxi_df['TotalAmount'] > 0]

taxi_df = taxi_df[taxi_df['Milage'] > 0]

taxi_df = taxi_df[taxi_df['Milage'] < 500]
taxi_df['FareAmount'].groupby(taxi_df['PROVIDER NAME']).sum()
# Change data format

taxi_df['StartDateTime'] = pd.to_datetime(taxi_df['StartDateTime'], errors='coerce')

taxi_df['EndDateTime'] = pd.to_datetime(taxi_df['EndDateTime'])



# Generate detailed columns of datetime

taxi_df['Date'] = taxi_df['StartDateTime'].dt.date

taxi_df['Hour'] = taxi_df['StartDateTime'].dt.hour

taxi_df['Weekday'] = taxi_df['StartDateTime'].dt.weekday

taxi_df['DayofWeek'] = taxi_df['StartDateTime'].dt.weekday_name
# Plot the heat map

time_map=pd.pivot_table(taxi_df,index=['DayofWeek'],columns =['Hour'],aggfunc='size')

fig, ax = plt.subplots(figsize=(20,15)) 

ax=sns.heatmap(time_map,linewidths=0.1,square=True,cmap='YlGnBu')
from mpl_toolkits.basemap import Basemap

from matplotlib import cm

west, south, east, north = -77.87, 38.00, -76.00, 39.5

fig = plt.figure(figsize=(14,10))

m = Basemap(projection='tmerc', llcrnrlat=south, urcrnrlat=north,

            llcrnrlon=west, urcrnrlon=east, lat_0=38.00,lon_0=-76.00, resolution='i')

m.drawstates()

x, y = m(taxi_df['DestinationLongitude'].values, taxi_df['DestinationLatitude'].values)

m.hexbin(x, y, gridsize=6000, bins='log',cmap=cm.YlOrRd_r)

m.colorbar(location='bottom')
# Classify weekdays and weekend

x = 0

taxi_df['IsWeekday'] = [x+1 if i<6 else x for i in taxi_df['Weekday']]
# Remove nulls and wrong values

taxi_df['StartDateTime'].dropna(inplace=True)

taxi_df = taxi_df[taxi_df['StartDateTime'] < taxi_df['EndDateTime']]
# Label Weekdays

df1 = DataFrame(taxi_df[taxi_df['IsWeekday']==1]['Hour'].value_counts().sort_index())

df1 = df1.reset_index()

df1.columns = ['Hour', 'Demand']



kmeans = KMeans(n_clusters=3, random_state=0).fit(df1)

df1['WeekdayLabel'] = kmeans.labels_
# Weekdays time tiers

df1
# Label Weekend

df2 = DataFrame(taxi_df[taxi_df['IsWeekday']==0]['Hour'].value_counts().sort_index())

df2 = df2.reset_index()

df2.columns = ['Hour', 'Demand']



kmeans = KMeans(n_clusters=3, random_state=0).fit(df2)

df2['WeekendLabel'] = kmeans.labels_
# Weekend time tiers

df2
# Merge labels

df1 = df1.loc[:,['Hour', 'WeekdayLabel']]

df2 = df2.loc[:,['Hour', 'WeekendLabel']]
taxi_df = taxi_df.merge(df1, on = 'Hour')
taxi_df = taxi_df.merge(df2, on = 'Hour')
taxi_df.columns
# Split dataset into weekdays data and weekend data

day_df = taxi_df[taxi_df['IsWeekday']==1].loc[:,['FareAmount', 'TotalAmount', 'IsWeekday', 'WeekdayLabel']]

end_df = taxi_df[taxi_df['IsWeekday']==0].loc[:,['FareAmount', 'TotalAmount', 'IsWeekday', 'WeekendLabel']]
# For weekdays

peak = day_df[day_df['WeekdayLabel']==1]

off_peak = day_df[day_df['WeekdayLabel']==0]
# For weekend

peak_end = end_df[end_df['WeekendLabel']==2]

off_peak_end = end_df[end_df['WeekendLabel']==1]
# Generate list of price shift and demand shift

elastic = 0.22

price_shift = np.linspace(0.01,2,200)

demand_shift = price_shift * elastic

shift = list(zip(price_shift, demand_shift))
# Weekday, Based on TotalAmount

rev_day = []

for x,y in shift:

    peak_samp = peak.sample(frac=(1-y), random_state=1)

    p = peak_samp['TotalAmount'].sum() * (1 + x)

    rev_day.append(p)
# Weekday Peak time, Result, Based on TotalAmount

rev_day = DataFrame(list(zip(price_shift, demand_shift, rev_day)), columns=['price_shift', 'demand_shift', 'revenue'])

peak_day_max = rev_day.sort_values(by='revenue', ascending=False).head(1)
peak_day_max['revenue shift']= max(rev_day['revenue'])/peak['TotalAmount'].sum()-1

peak_day_max
# Weekend, Based on TotalAmount

rev_end = []

for x,y in shift:

    peak_end_samp = peak_end.sample(frac=(1-y), random_state=1)

    p = peak_end_samp['TotalAmount'].sum() * (1 + x)

    rev_end.append(p)
# Weekend Peak time, Result, Based on TotalAmount

rev_end = DataFrame(list(zip(price_shift, demand_shift, rev_end)), columns=['price_shift', 'demand_shift', 'revenue'])

peak_end_max = rev_end.sort_values(by='revenue', ascending=False).head(1)
peak_end_max['revenue shift']= max(rev_end['revenue'])/peak_end['TotalAmount'].sum()-1

peak_end_max
# Weekdays, Generat a related list of demand shift, based on fare shift

elastic = 0.22

fare_shift = np.linspace(0.01,2,200)

demand_shift_day = []

for i in fare_shift:

    a = (sum(peak['TotalAmount'] + peak['FareAmount'] * i)/sum(peak['TotalAmount']) - 1)*elastic

    demand_shift_day.append(a)
#Weekday High Peak, Analysis, only fare

shift_day = list(zip(fare_shift, demand_shift))

rev_day2 = []

for x,y in shift_day:

    peak_samp2 = peak.sample(frac=(1-y), random_state=1)

    p = sum(peak_samp2['TotalAmount'] + peak_samp2['FareAmount'] * x)

    rev_day2.append(p)
# Weekday Peak time, Result, Based on FareAmount

rev_day2 = DataFrame(list(zip(fare_shift, demand_shift, rev_day2)), columns=['fare_shift', 'demand_shift', 'revenue'])

peak_day_max2 = rev_day2.sort_values(by='revenue', ascending=False).head(1)
peak_day_max2['revenue shift']= max(rev_day2['revenue'])/peak['TotalAmount'].sum()-1

peak_day_max2
# Weekend, Generat a related list of demand shift, based on fare shift

demand_shift_end = []

for i in fare_shift:

    a = (sum(peak_end['TotalAmount'] + peak_end['FareAmount'] * i)/sum(peak_end['TotalAmount']) - 1)*elastic

    demand_shift_end.append(a)
#Weekend High Peak, Analysis, Based on FareAmount

shift_end = list(zip(fare_shift, demand_shift_end))

rev_end2 = []

for x,y in shift_end:

    peak_end_samp2 = peak_end.sample(frac=(1-y), random_state=1)

    p = sum(peak_end_samp2['TotalAmount'] + peak_end_samp2['FareAmount'] * x)

    rev_end2.append(p)
# Weekend Peak time, Result, Based on FareAmount

rev_end2 = DataFrame(list(zip(fare_shift, demand_shift_end, rev_end2)), columns=['fare_shift', 'demand_shift', 'revenue'])

peak_end_max2 = rev_end2.sort_values(by='revenue', ascending=False).head(1)
peak_end_max2['revenue shift']= max(rev_end2['revenue'])/peak_end['TotalAmount'].sum()-1

peak_end_max2
# Generate list of price shift and demand shift

elastic = 0.22

price_shift_off = np.linspace(-0.01,-1,100)

demand_shift_off = price_shift_off * elastic

shift_off = list(zip(price_shift_off, demand_shift_off))
#Weekday Off-peak, Analysis

rev_off_day = []

for x,y in shift_off:

    off_samp = off_peak.sample(frac=(1+y), random_state=1, replace=True)

    p = off_samp['TotalAmount'].sum() * (1 + x)

    rev_off_day.append(p)
# Weekday Off-peak time, Result, Based on TotalAmount

rev_off_day = DataFrame(list(zip(price_shift_off, demand_shift_off, rev_off_day)), columns=['price_shift', 'demand_shift', 'revenue'])

off_day_max = rev_off_day.sort_values(by='revenue', ascending=False).head(1)
plt.scatter(-rev_off_day['demand_shift'], rev_off_day['revenue'])

plt.xlabel('Demand')

plt.ylabel('Revenue')

plt.title('Demand vs Revenue when Discount')

plt.show
#Weekend Off-peak, Analysis

rev_off_end = []

for x,y in shift_off:

    off_samp = off_peak_end.sample(frac=(1-y), random_state=1, replace=True)

    p = off_samp['TotalAmount'].sum() * (1 + x)

    rev_off_end.append(p)
# Weekday Off-peak time, Result, Based on TotalAmount

rev_off_end = DataFrame(list(zip(price_shift_off, demand_shift_off, rev_off_end)), columns=['price_shift', 'demand_shift', 'revenue'])

off_end_max = rev_off_end.sort_values(by='revenue', ascending=False).head(1)
plt.scatter(-rev_off_end['demand_shift'], rev_off_end['revenue'])

plt.xlabel('Demand')

plt.ylabel('Revenue')

plt.title('Demand vs Revenue when Discount')

plt.show
# Set supply of Peak Time

s_elastic = 0.28

price_shift = np.linspace(0.01,2,200)

supply_shift = price_shift * s_elastic

supply_day = (1+supply_shift) * len(peak)

supply_end = (1+supply_shift) * len(peak_end)
#Weekday Peak Time, Result, per supply

rev_day['supply'] = supply_day

rev_day['TotalAmount per Trip'] = rev_day['revenue']/rev_day['supply']

p_mean = peak['TotalAmount'].mean()

print('Original amount per trip(Weekday):', p_mean)

print('Max amount per trip(Weekday):', rev_day['TotalAmount per Trip'].max())

print('Price shift(Weekday):', rev_day['TotalAmount per Trip'].max()/p_mean -1)
#Weekend Peak Time, Result, per supply

rev_end['supply'] = supply_end

rev_end['TotalAmount per Trip'] = rev_end['revenue']/rev_end['supply']

print('Original amount per trip(Weekend):', peak_end['TotalAmount'].mean())

print('Max amount per trip(Weekend):', rev_end['TotalAmount per Trip'].max())

print('Price shift(Weekend):', rev_day['TotalAmount per Trip'].max()/peak_end['TotalAmount'].mean() -1)
# Set supply of Peak Time for Weekdays

supply_shift_day = np.array(demand_shift_day)  * (s_elastic/elastic)

supply_day2 = (1+supply_shift_day) * len(peak)
#Weekday Peak Time, Result, per supply

rev_day2['supply'] = supply_day2

rev_day2['TotalAmount per Trip'] = rev_day2['revenue']/rev_day['supply']

print('Original amount per trip(Weekday):', p_mean)

print('Max amount per trip(Weekday):', rev_day2['TotalAmount per Trip'].max())

print('Price shift(Weekday):', rev_day2['TotalAmount per Trip'].max()/p_mean -1)
# Set supply of Peak Time for Weekend

supply_shift_end = np.array(demand_shift_end)  * (s_elastic/elastic)

supply_end2 = (1+supply_shift_end) * len(peak_end)
#Weekend Peak Time, Result, per supply

rev_end2['supply'] = supply_end2

rev_end2['TotalAmount per Trip'] = rev_end2['revenue']/rev_end2['supply']

print('Original amount per trip(Weekend):', peak_end['TotalAmount'].mean())

print('Max amount per trip(Weekend):', rev_end2['TotalAmount per Trip'].max())

print('Price shift(Weekend):', rev_end2['TotalAmount per Trip'].max()/peak_end['TotalAmount'].mean() -1)