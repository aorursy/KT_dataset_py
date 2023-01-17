import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats
train = pd.read_csv('../input/train.csv')

print(train.shape)
print(train.head())
temp = train.copy()

temp = temp[temp.trip_duration < temp.trip_duration.quantile(0.995)] # Temporarily removing outliers

pickup_dates = pd.DatetimeIndex(temp['pickup_datetime'])
weekday = pickup_dates.dayofweek

day, count = np.unique(weekday, return_counts = True)



plt.figure(figsize=(6,4))

ax = sns.barplot(x = day, y = count)

ax.set(xlabel = "Day of week", ylabel = "Count of taxi rides")

plt.show();
hours = pickup_dates.hour

hour, count = np.unique(hours, return_counts = True)



plt.figure(figsize=(8,5))

ax = sns.barplot(x = hour, y = count)

ax.set(xlabel = "Hour of day", ylabel = "Count of taxi rides")

plt.show();
temp['Hour_of_day'] = hours

temp['Day_of_week'] = weekday

df_grp = temp.groupby(['Day_of_week', 'Hour_of_day'])['trip_duration']

sns.heatmap(df_grp.count().unstack())

plt.show()
sns.heatmap(df_grp.mean().unstack())

plt.show()
def haversine(lon1, lat1, lon2, lat2):

    """

    Calculate the great circle distance between two points 

    on the earth (specified in decimal degrees)

    """

    # convert decimal degrees to radians 

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula 

    dlon = lon2 - lon1 

    dlat = lat2 - lat1 

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2

    c = 2 * np.arcsin(np.sqrt(a)) 

    km = 6367 * c

    return km
temp['rad_dist'] = haversine(temp.pickup_longitude, temp.pickup_latitude, temp.dropoff_longitude, temp.dropoff_latitude)
trip_dur_grp = temp.groupby(['Day_of_week', 'Hour_of_day'])['rad_dist']

sns.heatmap(trip_dur_grp.mean().unstack())

plt.show()
slope, intercept, r_value, p_value, std_err = stats.linregress(temp.rad_dist, temp.trip_duration)
temp['linreg_dur_dist'] = np.add(intercept, np.multiply(slope, temp['rad_dist']))

temp['diff_vs_linreg'] = np.divide(temp['trip_duration'], temp['linreg_dur_dist'])
trip_dur_grp = temp.groupby(['Day_of_week', 'Hour_of_day'])['diff_vs_linreg']

sns.heatmap(trip_dur_grp.mean().unstack())

plt.show()
