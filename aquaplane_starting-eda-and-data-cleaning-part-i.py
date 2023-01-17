import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc
gc.enable()
train_df = pd.read_csv('../input/train.csv',nrows=10e6)
#parsing time - straight from https://www.kaggle.com/szelee/how-to-import-a-csv-file-of-55-million-rows
train_df['pickup_datetime'] = train_df['pickup_datetime'].str.slice(0, 16)
train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')
train_df.dtypes
train_df.isna().sum().plot('bar')
mss_long = train_df['dropoff_longitude'].isna()
mss_lat = train_df['dropoff_latitude'].isna()
print(f'Any missing values not occurring in pairs? {(mss_long!=mss_lat).any()}')
del mss_long, mss_lat
mss_df = train_df[train_df['dropoff_longitude'].isna()]
mss_df.head()
try:
    del mss_df
except NameError:
    pass
fares = train_df['fare_amount']
sns.boxplot(fares)
fares.sort_values(ascending=False)[:5]
plt.clf()
fares_capped = fares[fares<5000]
try:
    del fares
except NameError:
    pass
sns.boxplot(fares_capped)
plt.clf()
fares_capped.sort_values()[:5]
irreg_fares = train_df[(train_df['fare_amount']<2.5)]
display(irreg_fares.head())
print(f'there are {len(irreg_fares)} fare anomalies')
red_irreg_fares = train_df[(train_df['fare_amount']<2.5)&(train_df['fare_amount']>-2.5)]
print(f'there are {len(red_irreg_fares)} fare anomalies still remaining')
try:
    del irreg_fares, red_irreg_fares
except NameError:
    pass
pickup_times = train_df['pickup_datetime']
print(pickup_times.sort_values()[:10])
print(pickup_times.sort_values(ascending=False)[:10])
pickup_dates = pickup_times.dt.date
volume_series = pickup_dates.groupby(pickup_dates).count()
fig, ax = plt.subplots(1, 1, figsize = (10, 5))
sns.distplot(volume_series)
ax.set_xlabel('trip volume')
ax.set_ylabel('density')
plt.show()
plt.clf()
#in the pandas world, Monday=0 and Sunday=6, so anything <5 is a weekday
dayofweek = pd.to_datetime(pickup_dates).dt.dayofweek
weekday_pickup_dates = pickup_dates[dayofweek<5]
weekend_pickup_dates = pickup_dates[dayofweek>4]
weekday_volume_series = weekday_pickup_dates.groupby(weekday_pickup_dates).count()
weekend_volume_series = weekend_pickup_dates.groupby(weekend_pickup_dates).count()
fig, ax = plt.subplots(1, 1, figsize = (10, 5))
sns.distplot(weekday_volume_series)
ax.set_xlabel('weekday trip volume')
ax.set_ylabel('density')
plt.show()
plt.clf()
fig, ax = plt.subplots(1, 1, figsize = (10, 5))
sns.distplot(weekend_volume_series)
ax.set_xlabel('weekend trip volume')
ax.set_ylabel('density')
plt.show()
plt.clf()

del weekday_volume_series, weekend_volume_series
gc.collect()
fares = train_df['fare_amount'].copy()

#setting a lower cap right now to prevent distortion
fares[fares>1000] = 1000
mean_fares = fares.groupby(pickup_dates).mean()
print(f'Correlation between daily trip volume and mean fares is {mean_fares.corr(volume_series)}')