import pandas as pd

import matplotlib.pyplot as plt



%matplotlib inline

plt.style.use('ggplot')
df = pd.read_csv('../input/metro-bike-share-trip-data.csv', low_memory=False)

df.head(5)
ax = df['Passholder Type'].value_counts().plot.bar();

ax.set_title('Passholder type frequencies');
df['Passholder Type'].value_counts()
ax = (df.Duration/60).plot.hist(bins=30)

ax.set_title('Duration [min]');
from pandas.api.types import is_numeric_dtype



def remove_outlier(df):

    low = .05

    high = .95

    quant_df = df.quantile([low, high])

    for name in list(df.columns):

        if is_numeric_dtype(df[name]):

            df = df[(df[name] > quant_df.loc[low, name]) & (df[name] < quant_df.loc[high, name])]

    return df
df = remove_outlier(df)
ax = (df.Duration/60).plot.hist(bins=30)

ax.set_title('Duration [min]');
nr_route_cat = df['Trip Route Category'].count()

nr_one_way = len(df.loc[(df['Trip Route Category'] == 'One Way')])

nr_round_trip = len(df.loc[(df['Trip Route Category'] == 'Round Trip')])

print("Percentage of round trips: {}%".format(round(nr_round_trip / nr_route_cat * 100, 2)))
# Distance between pickup and dropoff



import numpy as np



start_lat = np.deg2rad(df['Starting Station Latitude'])

start_lon = np.deg2rad(df['Starting Station Longitude'])

stop_lat = np.deg2rad(df['Ending Station Latitude'])

stop_lon = np.deg2rad(df['Ending Station Longitude'])



dlon = stop_lon - start_lon

dlat = stop_lat - start_lat



a = np.sin(dlat / 2)**2 + np.cos(start_lat) * np.cos(stop_lat) * np.sin(dlon / 2)**2

c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

# approximate radius of earth in km

R = 6373.0

distance = R * c
distance_new = pd.DataFrame(distance)

distance_red = distance_new.loc[(distance_new[0] > 0)][0]
ax = distance_red.plot.hist(bins=30, title='One-way trip distances')

ax.set_xlabel('Distance [km]');
print("Trip duration summary statistics:")

print((df.Duration/60).describe())

print('\n')

print("Trip distance proxy summary statistics:")

print((distance_red).describe())
velocity_mean = (distance_red).mean() / (df.Duration/60).mean() * 60  # 60 factor converts from km/min to km/hr\

print("Mean velocity of riders: {} km/hr".format(round(velocity_mean,1)))

(distance_red / (df.Duration/60) * 60).plot.hist(bins=30)