import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns 
from mpl_toolkits.basemap import Basemap
%matplotlib inline
# Cities (lat, lon)
beijing = [40, 116.5]
london = [51.5, -0.25]
newyork = [40.75, -74]

# Set up Mercator projection plot for the whole world
plt.figure(1, figsize=(12,6))
m = Basemap(projection='merc',
            llcrnrlat=-60,
            urcrnrlat=65,
            llcrnrlon=-180,
            urcrnrlon=180,
            lat_ts=0,
            resolution='c')
#m.fillcontinents(color='#191919',lake_color='#000000') # dark grey land, black lakes
m.drawmapboundary(fill_color='#000000')                # black background
m.drawcountries(linewidth=0.1, color="w")              # thin white line for country borders

# Plot the cities
m.plot([beijing[1]-0.5,beijing[1]-0.5,beijing[1]+0.5,beijing[1]+0.5,beijing[1]-0.5],
       [beijing[0]-0.5,beijing[0]+0.5,beijing[0]+0.5,beijing[0]-0.5,beijing[0]-0.5],
       latlon=True)
m.plot([london[1]-3,london[1]-3,london[1]+3,london[1]+3,london[1]-3],
       [london[0]-3,london[0]+3,london[0]+3,london[0]-3,london[0]-3],
       latlon=True)
m.plot([newyork[1]-5,newyork[1]-5,newyork[1]+5,newyork[1]+5,newyork[1]-5],
       [newyork[0]-5,newyork[0]+5,newyork[0]+5,newyork[0]-5,newyork[0]-5],
       latlon=True)

plt.title("City locations")
plt.show()
df_events = pd.read_csv("../input/events.csv", dtype={'device_id': np.str})

idx_beijing = (df_events["longitude"]>beijing[1]-0.5) &\
              (df_events["longitude"]<beijing[1]+0.5) &\
              (df_events["latitude"]>beijing[0]-0.5) &\
              (df_events["latitude"]<beijing[0]+0.5)
df_events_beijing = df_events[idx_beijing]

idx_london =  (df_events["longitude"]>london[1]-3) &\
              (df_events["longitude"]<london[1]+3) &\
              (df_events["latitude"]>london[0]-3) &\
              (df_events["latitude"]<london[0]+3)
df_events_london = df_events[idx_london]

idx_newyork = (df_events["longitude"]>newyork[1]-5) &\
              (df_events["longitude"]<newyork[1]+5) &\
              (df_events["latitude"]>newyork[0]-5) &\
              (df_events["latitude"]<newyork[0]+5)
df_events_newyork = df_events[idx_newyork]

print("Total # events:", len(df_events))
print("Total # Beijing events:", len(df_events_beijing))
print("Total # London events:", len(df_events_london))
print("Total # New York events:", len(df_events_newyork))
plt.figure(1, figsize=(12,18))
plt.subplot(311)
plt.hist(df_events_newyork['timestamp'].map( lambda x: pd.to_datetime(x).hour ), bins=24)
plt.title("New York")
plt.subplot(312)
plt.hist(df_events_london['timestamp'].map( lambda x: pd.to_datetime(x).hour ), bins=24)
plt.title("London")
plt.subplot(313)
plt.hist(df_events_beijing['timestamp'].map( lambda x: pd.to_datetime(x).hour ), bins=24)
plt.title("Beijing")
plt.show()
plt.figure(1, figsize=(12,18))
plt.subplot(311)
plt.hist(df_events_newyork['timestamp'].map( lambda x: (pd.to_datetime(x).hour-12)%24 ), bins=24)
plt.title("New York")
plt.subplot(312)
plt.hist(df_events_london['timestamp'].map( lambda x: (pd.to_datetime(x).hour-7)%24 ), bins=24)
plt.title("London")
plt.subplot(313)
plt.hist(df_events_beijing['timestamp'].map( lambda x: pd.to_datetime(x).hour ), bins=24)
plt.title("Beijing")
plt.show()