import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

from matplotlib import animation

import numpy as np

import dateutil

from sklearn.cluster import MiniBatchKMeans

from ipywidgets import interact,  FloatSlider, RadioButtons

import geopandas

import matplotlib.pyplot as plt

import seaborn as sn

%matplotlib inline
import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv('../input/train.csv')
df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])

df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
df.head()
#filter dataset

xlim = [-74.03, -73.77]

ylim = [40.63, 40.90]

df = df[(df.pickup_longitude> xlim[0]) & (df.pickup_longitude < xlim[1])]

df = df[(df.dropoff_longitude> xlim[0]) & (df.dropoff_longitude < xlim[1])]

df = df[(df.pickup_latitude> ylim[0]) & (df.pickup_latitude < ylim[1])]

df = df[(df.dropoff_latitude> ylim[0]) & (df.dropoff_latitude < ylim[1])]

df = df[df.trip_duration < 5000]
df_byhour = df.groupby(df.pickup_datetime.dt.hour)["trip_duration"].mean()

df_bymonth = df.groupby(df.pickup_datetime.dt.month)["trip_duration"].mean()

df_byday = df.groupby(df.pickup_datetime.dt.weekday)["trip_duration"].mean()



df_heatmap_do = df.groupby([pd.cut(df.dropoff_latitude, np.arange(ylim[0], ylim[1], 0.01)),

                            pd.cut(df.dropoff_longitude, np.arange(xlim[0], xlim[1], 0.005))])["id"].count()

df_heatmap_pu = df.groupby([pd.cut(df.pickup_latitude, np.arange(ylim[0], ylim[1], 0.01)),

                            pd.cut(df.pickup_longitude, np.arange(xlim[0], xlim[1], 0.005))])["id"].count()

plot_df_dropoff = (df_heatmap_do.reset_index()

              .pivot(index='dropoff_latitude', columns='dropoff_longitude')).sort_index(ascending=False)

plot_df_pickup = (df_heatmap_pu.reset_index()

              .pivot(index='pickup_latitude', columns='pickup_longitude')).sort_index(ascending=False)
plt.figure(figsize=(20,20))



plt.subplot(2,2,1)

plt.title("Pick Up Plot")

plt.plot(df["pickup_longitude"],df["pickup_latitude"],'.',alpha = 0.4, markersize = 0.5)



plt.subplot(2,2,2)

plt.title("Pick Up Heatmap")

#plt.plot(df["dropoff_longitude"],df["dropoff_latitude"],'.',alpha = 0.4, markersize = 0.5,color = 'b')

#plt.pcolor(plot_df_pickup)

sn.heatmap(plot_df_pickup,xticklabels=False, yticklabels=False)



plt.subplot(2,2,3)

plt.title("Drop Off Plot")

plt.plot(df["dropoff_longitude"],df["dropoff_latitude"],'.',alpha = 0.4, markersize = 0.5)



plt.subplot(2,2,4)

plt.title("Drop Off Heatmap")

#plt.plot(df["dropoff_longitude"],df["dropoff_latitude"],'.',alpha = 0.4, markersize = 0.5,color = 'b')

sn.heatmap(plot_df_dropoff,xticklabels=False, yticklabels=False)



plt.show()
df_hours = df.groupby(df.pickup_datetime.dt.hour)["trip_duration"].mean()
fig, ax = plt.subplots(1, figsize = (10,10))

plt.bar(df_hours.index.values, df_hours, align = "center")

rect_morn = matplotlib.patches.Rectangle((7,0), 2, 1000, angle=0, fill = True, 

                                    linewidth = 2.0, edgecolor = 'r', alpha = 0.3, linestyle = '--')

rect_eve = matplotlib.patches.Rectangle((16,0), 2, 1000, angle=0, fill = True, 

                                    linewidth = 2.0, edgecolor = 'r', alpha = 0.3, linestyle = '--')

ax.add_patch(rect_morn)

ax.add_patch(rect_eve)

plt.title("Trip Duration by Hour")

plt.xlabel("Hour")

plt.ylabel("Average Trip Duration")

plt.ylim([400,1000])

plt.show()
count_byhour = df.groupby(df.pickup_datetime.dt.hour)["id"].count()

count_bymonth = df.groupby(df.pickup_datetime.dt.month)["id"].count()

count_byday = df.groupby(df.pickup_datetime.dt.weekday)["id"].count()



plt.figure(figsize=(20,17))

plt.subplot(321)

plt.plot(df_byhour)

plt.title("Average Ride Duration per Hour (Seconds)")

plt.ylabel("Average Ride Duration")

plt.xlabel("Hour")

plt.subplot(322)

plt.plot(count_byhour)

plt.title("Ride Volume per Hour")

plt.ylabel("Ride Volume")

plt.xlabel("Hour")



plt.subplot(325)

plt.plot(df_bymonth)

plt.title("Average Ride Duration per Month (Seconds)")

plt.ylabel("Average Ride Duration")

plt.xlabel("Month")

plt.subplot(326)

plt.plot(count_bymonth)

plt.title("Ride Volume per Month")

plt.ylabel("Ride Volume")

plt.xlabel("Month")



plt.subplot(323)

plt.plot(df_byday)

plt.xlabel("Day")

plt.ylabel("Average Ride Duration (Seconds)")

plt.title("Average Ride Duration per Day")

plt.subplot(324)

plt.plot(count_byday)

plt.ylabel("Ride Volume")

plt.title("Ride Volume per Day")

plt.xlabel("Day")
by_passenger_count = df.groupby(df.passenger_count)["trip_duration"].mean()

plt.figure(figsize=(10,10))

plt.title("Passenger Count vs Average Duration")

plt.xlabel("Number of Passagers")

plt.ylabel("Average Duration")

plt.bar(by_passenger_count.index.values, by_passenger_count, align = "center")
def haversine_array(lat1, lng1, lat2, lng2):

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    AVG_EARTH_RADIUS = 6371  # in km

    lat = lat2 - lat1

    lng = lng2 - lng1

    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2

    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))

    return h



def dummy_manhattan_distance(lat1, lng1, lat2, lng2):

    a = haversine_array(lat1, lng1, lat1, lng2)

    b = haversine_array(lat1, lng1, lat2, lng1)

    return a + b



def bearing_array(lat1, lng1, lat2, lng2):

    AVG_EARTH_RADIUS = 6371  # in km

    lng_delta_rad = np.radians(lng2 - lng1)

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    y = np.sin(lng_delta_rad) * np.cos(lat2)

    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)

    return np.degrees(np.arctan2(y, x))
df.loc[:, 'distance_haversine'] = haversine_array(df['pickup_latitude'].values, df['pickup_longitude'].values, df['dropoff_latitude'].values, df['dropoff_longitude'].values)

df.loc[:, 'distance_dummy_manhattan'] =  dummy_manhattan_distance(df['pickup_latitude'].values, df['pickup_longitude'].values, df['dropoff_latitude'].values, df['dropoff_longitude'].values)

df.loc[:, 'direction'] = bearing_array(df['pickup_latitude'].values, df['pickup_longitude'].values, df['dropoff_latitude'].values, df['dropoff_longitude'].values)
df_heatmap_distance = df.groupby([pd.cut(df.dropoff_latitude, np.arange(ylim[0], ylim[1], 0.01)),

                            pd.cut(df.dropoff_longitude, np.arange(xlim[0], xlim[1], 0.005))])["distance_dummy_manhattan"].mean()

plot_df_distance = (df_heatmap_distance.reset_index()

              .pivot(index='dropoff_latitude', columns='dropoff_longitude')).sort_index(ascending=False)
plt.figure(figsize=(10,10))

sn.heatmap(plot_df_distance ,xticklabels=False, yticklabels=False)

plt.title("Heatmap of Average Distance Traveled")

plt.xlabel("Longitude")

plt.ylabel("Latitude")
coords = np.vstack((df[['pickup_latitude', 'pickup_longitude']].values,

                    df[['dropoff_latitude', 'dropoff_longitude']].values))

sample_ind = np.random.permutation(len(coords))[:500000]

kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])



df.loc[:, 'pickup_cluster'] = kmeans.predict(df[['pickup_latitude', 'pickup_longitude']])

df.loc[:, 'dropoff_cluster'] = kmeans.predict(df[['dropoff_latitude', 'dropoff_longitude']])
plt.figure(figsize=(10,10))

plt.scatter(df.pickup_longitude,df.pickup_latitude, 

            c = df.pickup_cluster, cmap = "nipy_spectral",marker = '.', alpha = 0.4)

plt.grid(False)

plt.title("")

plt.show()
df_plot = df[(df.pickup_longitude> -74.0) & (df.pickup_longitude < -73.95)]

df_plot = df[(df.pickup_latitude> 40.725) & (df.pickup_latitude < 40.8)]
ave_cluster_loc = df.groupby("pickup_cluster").mean()

df["cluster_pair"] = list(zip(df.pickup_cluster, df.dropoff_cluster))

top_pairs_df = df.groupby("cluster_pair")["id"].count().sort_values(axis = 0, ascending = False)

top_pairs_index = top_pairs_df.index.values[:5]
top_pairs_df.head()
def plotit(ind = 0):



    fig, ax = plt.subplots(figsize=(11,11))

    ax.scatter(df_plot.pickup_longitude,df_plot.pickup_latitude, 

                c = df_plot.pickup_cluster, cmap = "nipy_spectral",marker = '.', alpha = 0.01)

    ax.scatter(ave_cluster_loc["pickup_longitude"],ave_cluster_loc["pickup_latitude"],marker = '.')



    for index, row in ave_cluster_loc.iterrows():



           ax.annotate(index, (row["pickup_longitude"],row["pickup_latitude"]))

    

    ax.arrow(ave_cluster_loc.loc[top_pairs_index[ind][0]].pickup_longitude, 

             ave_cluster_loc.loc[top_pairs_index[ind][0]].pickup_latitude,

            ave_cluster_loc.loc[top_pairs_index[ind][1]].pickup_longitude - ave_cluster_loc.loc[top_pairs_index[ind][0]].pickup_longitude - 0.004, 

            ave_cluster_loc.loc[top_pairs_index[ind][1]].pickup_latitude - ave_cluster_loc.loc[top_pairs_index[ind][0]].pickup_latitude -0.004,

     head_width = 0.005, head_length = 0.005)  



    ax.set_xlim([-74.0,-73.95]) 

    ax.set_ylim([40.725, 40.8])
interact(plotit, ind=(0,4,1))