import time

import numpy as np

import pandas as pd

import seaborn as sns



import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

from matplotlib import cm

import mplleaflet as mpl

%matplotlib inline



from sklearn.cluster import DBSCAN

from geopy.distance import great_circle

from shapely.geometry import MultiPoint
rides = pd.read_csv('../input/uber-raw-data-aug14.csv')

rides.info()
rides.head()
rides.columns = ['timestamp', 'lat', 'lon', 'base']
ti = time.time()



rides['timestamp'] = pd.to_datetime(rides['timestamp'])



tf = time.time()

print(tf-ti,' seconds.')
rides.to_pickle('./test_data.pkl')

# rides = pd.read_pickle('./test_data.pkl')
rides['weekday'] = rides.timestamp.dt.weekday_name

rides['month'] = rides.timestamp.dt.month

rides['day'] = rides.timestamp.dt.day

rides['hour'] = rides.timestamp.dt.hour

rides['minute'] = rides.timestamp.dt.minute



## customized features

# rides['month_name'] = rides.timestamp.dt.strftime('%B')

# rides['day_hour'] = rides.timestamp.dt.strftime('%d-%H')



## ocular analysis

rides.head()
day_map = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday', 'Sunday']



rides['weekday'] = pd.Categorical(rides['weekday'], categories=day_map, ordered=True)
## groupby operation

hourly_ride_data = rides.groupby(['day','hour','weekday'])['timestamp'].count()



## reset index

hourly_ride_data = hourly_ride_data.reset_index()



## rename column

hourly_ride_data = hourly_ride_data.rename(columns = {'timestamp':'ride_count'})



## ocular analysis

hourly_ride_data.head()
## groupby operation

weekday_hourly_avg = hourly_ride_data.groupby(['weekday','hour'])['ride_count'].mean()



## reset index

weekday_hourly_avg = weekday_hourly_avg.reset_index()



## rename column

weekday_hourly_avg = weekday_hourly_avg.rename(columns = {'ride_count':'average_rides'})



## sort by categorical index

weekday_hourly_avg = weekday_hourly_avg.sort_index()



## ocular analysis

weekday_hourly_avg.head()
tableau_color_blind = [(0, 107, 164), (255, 128, 14), (171, 171, 171), (89, 89, 89),

             (95, 158, 209), (200, 82, 0), (137, 137, 137), (163, 200, 236),

             (255, 188, 121), (207, 207, 207)]



for i in range(len(tableau_color_blind)):  

    r, g, b = tableau_color_blind[i]  

    tableau_color_blind[i] = (r / 255., g / 255., b / 255.)
## create figure

fig = plt.figure(figsize=(12,6))

ax = fig.add_subplot(111)



## set palette   

current_palette = sns.color_palette(tableau_color_blind)



## plot data

sns.pointplot(ax=ax, x='hour',y='average_rides',hue='weekday', 

              palette = current_palette, data = weekday_hourly_avg)



## clean up the legend

l = ax.legend()

l.set_title('')



## format plot labels

ax.set_title('Weekday Averages for August 2014', fontsize=30)

ax.set_ylabel('Rides per Hour', fontsize=20)

ax.set_xlabel('Hour', fontsize=20)
def heat_map(ax_loc,title_str,rides_this_hour,nsew):

    

    ## get the axis

    ax = fig.add_subplot(ax_loc)



    ## make the basemap object

    m = Basemap(projection='merc', urcrnrlat=nsew[0], llcrnrlat=nsew[1],

                urcrnrlon=nsew[2], llcrnrlon=nsew[3], lat_ts=nsew[1], resolution='f')



    ## draw the background features

    m.drawmapboundary(fill_color = 'xkcd:light blue')

    m.fillcontinents(color='xkcd:grey', zorder = 1)

    m.drawcoastlines()

    m.drawrivers()



    ## project the GPS coordinates onto the x,y representation

    x, y = m(rides_this_hour['lon'].values, rides_this_hour['lat'].values)



    ## count the instances using the hexbin method and plot the results

    m.hexbin(x, y, gridsize=1000, mincnt = 1, bins = 'log', cmap=cm.YlOrRd, zorder = 2);



    ## set the title

    ax.set_title(title_str, fontsize=24)
## set weekday for analysis

target_day = 'Thursday'



## north,south,east,west lat/lon coordinates for bounding box

nsew = [40.9, 40.6, -73.8, -74.1]



## create figure

fig = plt.figure(figsize=(14,8))



## target hours

hrs = [17, 21, 23]



## axis subplot locations

ax_loc = [131, 132, 133] 



## title strings

title_str = ['5 PM', '9 PM', '11 PM']



## plot loop

for ii in range(len(ax_loc)):



    ## get the ride data from the target hour

    rides_this_hour = rides.loc[(rides['weekday'] == target_day) & (rides['hour'] == hrs[ii])]



    ##  plot the heat map

    heat_map(ax_loc[ii],title_str[ii],rides_this_hour,nsew)
hourly_ride_data.head()
thursday_hourly_data = hourly_ride_data[hourly_ride_data['weekday']=='Thursday']
## create figure

fig = plt.figure(figsize=(12,6))

ax = fig.add_subplot(111)



## set palette   

current_palette = sns.color_palette(tableau_color_blind)



## plot data

sns.pointplot(ax=ax, x='hour',y='ride_count',hue='day', palette = current_palette, data = thursday_hourly_data)



## clean up the legend

l = ax.legend()

l.set_title('')



## format plot labels

ax.set_title('Hourly Ride Count for Thursdays in August 2014', fontsize=25)

ax.set_ylabel('Rides', fontsize=20)

ax.set_xlabel('Hour', fontsize=20)
## set day for analysis

target_day = [14, 21]



## north,south,east,west lat/lon coordinates for bounding box

nsew = [40.9,40.6,-73.8,-74.1]



## create figure

fig = plt.figure(figsize=(14,8))



## hour 

hrs = 22



## axis locations

ax_loc = [121, 122] 



## title strings

title_str = ['Aug 14, 10 PM', 'Aug 21, 10 PM']



## plot loop

for ii in range(len(ax_loc)):



    ## get the ride data from the target hour

    rides_this_hour = rides.loc[(rides['day'] == target_day[ii]) & (rides['hour'] == hrs)]



    ## plot the heat map

    heat_map(ax_loc[ii],title_str[ii],rides_this_hour,nsew)
## make the figure

fig = plt.figure(figsize=(8,8))

ax = fig.add_subplot(111)



## get ride data

rides_this_hour = rides.loc[(rides['day']== 14) & (rides['hour'] == 22) & (rides['minute'] < 16)]



## plot ride data

plt.plot(rides_this_hour['lon'], rides_this_hour['lat'], 'bo', markersize=4)



## display the Leaflet

# mpl.show()     # opens in a new interactive tab

mpl.display()  # shows interactive map inline in Jupyter but cannot handle large data sets
def get_hot_spots(max_distance,min_cars,ride_data):

    

    ## get coordinates from ride data

    coords = ride_data.as_matrix(columns=['lat', 'lon'])

    

    ## calculate epsilon parameter using

    ## the user defined distance

    kms_per_radian = 6371.0088

    epsilon = max_distance / kms_per_radian

    

    ## perform clustering

    db = DBSCAN(eps=epsilon, min_samples=min_cars,

                algorithm='ball_tree', metric='haversine').fit(np.radians(coords))

    

    ## group the clusters

    cluster_labels = db.labels_

    num_clusters = len(set(cluster_labels))

    clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])

    

    ## report

    print('Number of clusters: {}'.format(num_clusters))

    

    ## initialize lists for hot spots

    lat = []

    lon = []

    num_members = []

    

    ## loop through clusters and get centroids, number of members

    for ii in range(len(clusters)):



        ## filter empty clusters

        if clusters[ii].any():



            ## get centroid and magnitude of cluster

            lat.append(MultiPoint(clusters[ii]).centroid.x)

            lon.append(MultiPoint(clusters[ii]).centroid.y)

            num_members.append(len(clusters[ii]))

            

    hot_spots = [lon,lat,num_members]

    

    return hot_spots
## get ride data

ride_data = rides.loc[(rides['day']== 21) & (rides['hour'] > 15)]



## maximum distance between two cluster members in kilometers

max_distance = 0.05



## minimum number of cluster members

min_pickups = 25



## call the get_hot_spots function

hot_spots = get_hot_spots(max_distance ,min_pickups, ride_data)
## make the figure

fig = plt.figure(figsize=(14,8))

ax = fig.add_subplot(111)



## set the color scale

color_scale = np.log(hot_spots[2])

# color_scale = hot_spots[2]



## make the scatter plot

plt.scatter(hot_spots[0], hot_spots[1],s=80,c=color_scale,cmap=cm.cool)



## display the Leaflet

# mpl.show()     # opens in a new interactive tab

mpl.display()  # shows interactive map inline in Jupyter but cannot handle large data sets