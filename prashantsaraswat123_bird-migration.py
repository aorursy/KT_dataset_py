# Import necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# Read the data
birthdata = pd.read_csv('../input/bird_tracking.csv')
birthdata.head(15)
birthdata.info()
# find unique birds
bird_names = pd.unique(birthdata['bird_name'])
plt.figure(figsize=(8,8))
for bird_name in bird_names:
    ix = birthdata['bird_name'] == bird_name
    x, y = birthdata.longitude[ix], birthdata.latitude[ix]
    plt.plot(x, y, '.', label=bird_name)
sns.set_style('whitegrid')    
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(loc='upper left')
# find eric speed
plt.figure(figsize=(7, 7))
speed = birthdata[birthdata['bird_name'] == 'Eric']['speed_2d']
ind = np.isnan(speed)
plt.hist(speed[~ind], bins = np.linspace(0, 30, 20))
plt.xlabel('2D speed (m/s)')
plt.ylabel('Frequency')
# find nico speed
plt.figure(figsize=(7, 7))
speed = birthdata[birthdata['bird_name'] == 'Nico']['speed_2d']
ind = np.isnan(speed)
plt.hist(speed[~ind], bins = np.linspace(0, 30, 20))
plt.xlabel('2D speed (m/s)')
plt.ylabel('Frequency');
# Create timestamp column
timestamps = []
for k in range(len(birthdata)):
    timestamps.append(datetime.datetime.strptime(birthdata.date_time.iloc[k][:-3], '%Y-%m-%d %H:%M:%S'))

birthdata['timestamp'] = pd.Series(timestamps, index=birthdata.index)
birthdata.head(15)
# Eric bird timestamp
data = birthdata[birthdata['bird_name'] == 'Eric']
times = birthdata.timestamp[birthdata['bird_name'] == 'Eric']
elapsed_time = [time - times[0] for time in times]
elapsed_days = np.array(elapsed_time) / datetime.timedelta(days=1)

next_day = 1
inds = []
daily_mean_speed = []

for i, t in enumerate(elapsed_days):
    if t < next_day:
        inds.append(i)
    else:
        daily_mean_speed.append(np.mean(data.speed_2d[inds]))
        next_day += 1
        inds = []
        
plt.figure(figsize=(10, 10))
plt.plot(daily_mean_speed)
plt.xlabel('Day')
plt.ylabel('Mean speed (m/s)')
proj = ccrs.Mercator()

plt.figure(figsize=(10, 10))
ax = plt.axes(projection=proj)
ax.set_extent((-25.0, 20.0, 52.0, 10.0))
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')

for names in bird_names:
    ix = birthdata['bird_name'] == names
    x, y = birthdata.longitude[ix], birthdata.latitude[ix]
    ax.plot(x, y, '.', transform=ccrs.Geodetic(), label=names)
    plt.legend(loc='upper left')
