import numpy as np
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns 
from mpl_toolkits.basemap import Basemap
%matplotlib inline
# Load
df_events = pd.read_csv("../input/events.csv", parse_dates=["timestamp"])
print("Total # events:", len(df_events))

# Transform
df_events["dayofweek"] = df_events["timestamp"].map(lambda x : x.dayofweek)
df_events["hour"] = df_events["timestamp"].map(lambda x : x.hour)
df_events["minute"] = df_events["timestamp"].map(lambda x : x.minute)

# Filter
idx_china = (df_events["longitude"]>75) &\
            (df_events["longitude"]<135) &\
            (df_events["latitude"]>15) &\
            (df_events["latitude"]<55)
df_events_china = df_events[idx_china]
print("Number of China events:", len(df_events_china))

idx_friday_night = (df_events_china["dayofweek"]==4) &\
                   (df_events_china["hour"]<6)
df_friday_night = df_events_china[idx_friday_night]
print("Number of Friday night events:", len(df_friday_night))

df_people = pd.read_csv("../input/gender_age_train.csv")
df_joined = pd.merge(df_people, df_friday_night, on="device_id", how="inner")
df_joined["group"].value_counts()
plt.figure(1, figsize=(12,6))

df_f = df_joined[df_joined["gender"] == "F"]
plt.subplot(121)
m1 = Basemap(projection='merc',
            llcrnrlat=15, urcrnrlat=55, llcrnrlon=75, urcrnrlon=135,
            lat_ts=35, resolution='i')
m1.drawmapboundary(fill_color='#000000')                # black background
m1.drawcountries(linewidth=0.1, color="w")              # thin white line for country borders
mxy1 = m1(df_f["longitude"].tolist(), df_f["latitude"].tolist())
m1.scatter(mxy1[0], mxy1[1], s=5, c="#db92db", lw=0, alpha=0.05, zorder=5)
plt.title("China's Female Night Owls")

df_m = df_joined[df_joined["gender"] == "M"]
plt.subplot(122)
m2 = Basemap(projection='merc',
            llcrnrlat=15, urcrnrlat=55, llcrnrlon=75, urcrnrlon=135,
            lat_ts=35, resolution='i')
m2.drawmapboundary(fill_color='#000000')                # black background
m2.drawcountries(linewidth=0.1, color="w")              # thin white line for country borders
mxy2 = m2(df_m["longitude"].tolist(), df_m["latitude"].tolist())
m2.scatter(mxy2[0], mxy2[1], s=5, c="#1292db", lw=0, alpha=0.05, zorder=5)
plt.title("China's Male Night Owls")


plt.show()
plt.figure(1, figsize=(12,12))

df_f23 = df_joined[df_joined["group"] == "F23-"]
plt.subplot(221)
m1 = Basemap(projection='merc',
            llcrnrlat=15, urcrnrlat=55, llcrnrlon=75, urcrnrlon=135,
            lat_ts=35, resolution='i')
m1.drawmapboundary(fill_color='#000000')                # black background
m1.drawcountries(linewidth=0.1, color="w")              # thin white line for country borders
mxy1 = m1(df_f23["longitude"].tolist(), df_f23["latitude"].tolist())
m1.scatter(mxy1[0], mxy1[1], s=5, c="#db92db", lw=0, alpha=0.05, zorder=5)
plt.title("China's F23- Night Owls")

df_f43 = df_joined[df_joined["group"] == "F43+"]
plt.subplot(222)
m2 = Basemap(projection='merc',
            llcrnrlat=15, urcrnrlat=55, llcrnrlon=75, urcrnrlon=135,
            lat_ts=35, resolution='i')
m2.drawmapboundary(fill_color='#000000')                # black background
m2.drawcountries(linewidth=0.1, color="w")              # thin white line for country borders
mxy2 = m2(df_f43["longitude"].tolist(), df_f43["latitude"].tolist())
m2.scatter(mxy2[0], mxy2[1], s=5, c="#db92db", lw=0, alpha=0.05, zorder=5)
plt.title("China's F43+ Night Owls")

df_m22 = df_joined[df_joined["group"] == "M22-"]
plt.subplot(223)
m3 = Basemap(projection='merc',
            llcrnrlat=15, urcrnrlat=55, llcrnrlon=75, urcrnrlon=135,
            lat_ts=35, resolution='i')
m3.drawmapboundary(fill_color='#000000')                # black background
m3.drawcountries(linewidth=0.1, color="w")              # thin white line for country borders
mxy3 = m3(df_m22["longitude"].tolist(), df_m22["latitude"].tolist())
m3.scatter(mxy3[0], mxy3[1], s=5, c="#1292db", lw=0, alpha=0.05, zorder=5)
plt.title("China's M22- Night Owls")

df_m39 = df_joined[df_joined["group"] == "M39+"]
plt.subplot(224)
m4 = Basemap(projection='merc',
            llcrnrlat=15, urcrnrlat=55, llcrnrlon=75, urcrnrlon=135,
            lat_ts=35, resolution='i')
m4.drawmapboundary(fill_color='#000000')                # black background
m4.drawcountries(linewidth=0.1, color="w")              # thin white line for country borders
mxy4 = m4(df_m39["longitude"].tolist(), df_m39["latitude"].tolist())
m4.scatter(mxy4[0], mxy4[1], s=5, c="#1292db", lw=0, alpha=0.05, zorder=5)
plt.title("China's M39+ Night Owls")

plt.show()