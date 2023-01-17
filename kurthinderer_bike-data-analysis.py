#standard library set up

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib

import seaborn as sns

import math



%matplotlib inline
# I will be making maps and due to the scale, they're going to need to be higher res with Basemap

!yes Y | conda install -c conda-forge basemap-data-hires

from mpl_toolkits.basemap import Basemap
path = '/kaggle/input/bikedata/'

bike_df = pd.read_csv(path + 'bike.csv')

station_df = pd.read_csv(path + 'stations.csv')

bike_df.head()
station_df.head()
bike_df['start_time'] = pd.to_datetime(bike_df['start_time'])

bike_df['end_time'] = pd.to_datetime(bike_df['end_time'])

bike_df.dtypes
# start by reexamining the durrations of times.

print('Number 1 second or more: ' + str(bike_df[bike_df['duration_sec'] > 0].shape[0]))

print('Number 1 minute or more: ' + str(bike_df[bike_df['duration_sec'] > 60].shape[0]))

print('Number 1 hour or more: ' + str(bike_df[bike_df['duration_sec'] > 3600].shape[0]))

print('Number 1 day or more: ' + str(bike_df[bike_df['duration_sec'] > 86400].shape[0]))
labels = np.array([0, 15, 30, 45, 60, 75, 90])

plt.hist(bike_df['duration_sec'], bins = 90, range = (0, 5400))

plt.title('Ride Duration')

plt.xticks(ticks = labels*60, labels = labels)

plt.xlabel('Duration (minutes)')

plt.ylabel('Count');
ticks = [1, 1+1/3, 1+2/3, 2, 2+1/3, 2+2/3, 3]

labels = ['1 min', '4 min', '15 min', '1 hr', '4 hrs', '15 hrs', '']

plt.hist(np.log(bike_df['duration_sec'])/np.log(60), bins = 40)

plt.xticks(ticks = ticks, labels = labels)

plt.title('Ride Duration (logarithmic scale)')

plt.ylabel('Count')

plt.xlabel('Duration')

plt.savefig('duration.png');
color = sns.color_palette()[0]

sns.countplot(data = bike_df, x = 'user_type', color = color)

plt.title('Number of Customers and Subscribers')

plt.yticks(ticks = [0, 500000, 1000000, 1500000, 2000000, 2500000], labels = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5])

plt.ylabel('Count (in millions)')

plt.xlabel('User Type')
plt.figure(figsize = (10,5))

plt.subplot(1,2,1)

sns.countplot(x = (bike_df['start_station_id'] == -1), color = color)

plt.title('Pick-up Location')

plt.yticks(ticks = [0, 500000, 1000000, 1500000, 2000000, 2500000, 3000000], 

           labels = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

plt.xticks(ticks = [False, True], labels = ['Station', 'Non-Station'])

plt.ylabel('Count (in millions)')

plt.xlabel('Location')



plt.subplot(1,2,2)

sns.countplot(x = (bike_df['end_station_id'] == -1), color = color)

plt.title('Drop-off Location')

plt.yticks(ticks = [0, 500000, 1000000, 1500000, 2000000, 2500000, 3000000], 

           labels = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

plt.xticks(ticks = [False, True], labels = ['Station', 'Non-Station'])

plt.ylabel('Count (in millions)')

plt.xlabel('Location');
bike_use_count = bike_df['bike_id'].value_counts()

bike_use_count.describe()
sns.violinplot(y = bike_use_count)

plt.title('Number of Times a Bike is Used')

plt.ylabel('Times Used')
start_station_count = bike_df[bike_df['start_station_id'] > 0]['start_station_id'].value_counts()

end_station_count = bike_df[bike_df['end_station_id'] > 0]['end_station_id'].value_counts()

start_station_count.describe(), end_station_count.describe()
plt.figure(figsize = (12,5))

plt.subplot(1,2,1)

sns.violinplot(y = start_station_count)

plt.title('Station Pick-up Count')

plt.ylabel('Number of Pick-ups')

plt.ylim((-8000,68000))



plt.subplot(1,2,2)

sns.violinplot(y = end_station_count)

plt.title('Station Drop-off Count')

plt.ylabel('Number of Drop-offs')

plt.ylim((-8000,68000));
start_times = pd.Series(data = 1, index = bike_df['start_time'])

plt.plot(start_times.resample('W').count())

plt.title('Weekly Bike Usage')

plt.ylim(0,120000)

plt.ylabel('Count')

plt.xlabel('Time');
day_count = start_times.resample('D').count()

day_of_week_mean = day_count.groupby(day_count.index.dayofweek).mean()

plt.bar(day_of_week_mean.index, day_of_week_mean)

plt.xticks(ticks = [0, 1, 2, 3, 4, 5, 6], labels = ['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun'])

plt.title('Mean Rides by Day of the Week')

plt.xlabel('Day')

plt.ylabel('Mean');
hour_count = start_times.resample('H').count()

hour_of_day_mean = hour_count.groupby(hour_count.index.time).mean()



plt.figure(figsize = (14,4))

plt.bar(['12am', '1am', '2am', '3am', '4am', '5am', '6am', '7am',

        '8am', '9am', '10am', '11am', '12pm', '1pm', '2pm', '3pm',

        '4pm', '5pm', '6pm', '7pm', '8pm', '9pm', '10pm', '11pm'], hour_of_day_mean)

plt.title('Mean Rides by Hour of the Day')

plt.xlabel('Hour')

plt.ylabel('Mean');
# This gets the basics for the maps I'm going to use. The box coordinates is a list of ordered pairs for 

# lower left and upper right. I'm going to need some city level maps so I have rhe resolution at full, 

# but for the entire bay area, I'm going to be using high.

def map_plot(box_coord, res = 'f'):

    m = Basemap(projection='cyl', llcrnrlat = box_coord[0][0], urcrnrlat = box_coord[1][0], 

                llcrnrlon = box_coord[0][1], urcrnrlon = box_coord[1][1], resolution=res)

    m.fillcontinents(zorder = 0)

    m.drawmapboundary()

    m.drawcoastlines()

    return m
# Getting the lower left and upper right for the bay area (min and max coordinate for the table)

# I'm using the end points of the rides as I would assume that it would be the furthest appart

# I'm adding a little bit of standard deviation so no points are on the end

bay_coord= [(bike_df['end_station_latitude'].min() - 0.2* bike_df['end_station_latitude'].std(),

              bike_df['end_station_longitude'].min() - 0.2* bike_df['end_station_longitude'].std()),

             (bike_df['end_station_latitude'].max() + 0.2* bike_df['end_station_latitude'].std(),

             bike_df['end_station_longitude'].max() + 0.2* bike_df['end_station_longitude'].std())]

bay_coord
plt.figure(figsize = (8,8))

bay_map = map_plot(bay_coord, res = 'h')

bay_map.scatter(station_df['station_longitude'], station_df['station_latitude'], 

          latlon = True, color = 'b', zorder = 10, s = 3, alpha = 0.4)

plt.title('Map of Stations')
plt.hist(station_df['station_longitude'], bins = 25)

plt.title('Station Longitudes')
#get the min, max, and center coordinates for each region

def min_max_cen (df):

    min_coord = (df['station_latitude'].min() - df['station_latitude'].std(), 

                 df['station_longitude'].min() - df['station_longitude'].std())

    max_coord = (df['station_latitude'].max() + df['station_latitude'].std(), 

                 df['station_longitude'].max() + df['station_longitude'].std())

    center_coord = ((max_coord[0]+min_coord[0])/2, (max_coord[1]+min_coord[1])/2)

    return min_coord, max_coord, center_coord



sf_st_df = station_df[station_df['station_longitude'] <= -122.35]

oak_st_df = station_df[(station_df['station_longitude'] > -122.35)&(station_df['station_longitude'] <= -122.1)]

sj_st_df = station_df[station_df['station_longitude'] > -122.1]



sf_min, sf_max, sf_cen = min_max_cen(sf_st_df)

oak_min, oak_max, oak_cen = min_max_cen(oak_st_df)

sj_min, sj_max, sj_cen = min_max_cen(sj_st_df)



sf_cen, oak_cen, sj_cen
# Just using the min and max as lower and upper bounds as boundrys, the boxes are not a uniform size.

# To do that I'm going to use the min and max of the of the bounding boxes for all of the boxes.

# And that was a bit rambling...

lat_max = max([sf_max[0] - sf_min[0], oak_max[0] - oak_min[0], sj_max[0] - sj_min[0]])

lon_max = max([sf_max[1] - sf_min[1], oak_max[1] - oak_min[1], sj_max[1] - sj_min[1]])



sf_coord = [(sf_cen[0] - lat_max/2, sf_cen[1] - lon_max/2), (sf_cen[0] + lat_max/2, sf_cen[1] + lon_max/2)]

oak_coord = [(oak_cen[0] - lat_max/2, oak_cen[1] - lon_max/2), (oak_cen[0] + lat_max/2, oak_cen[1] + lon_max/2)]

sj_coord = [(sj_cen[0] - lat_max/2, sj_cen[1] - lon_max/2), (sj_cen[0] + lat_max/2, sj_cen[1] + lon_max/2)]



sf_coord, oak_coord, sj_coord, lat_max, lon_max
cities = [(sf_coord,'San Francisco'),(oak_coord, 'Oakland'),(sj_coord, 'San Jose')]



# Now that I have that, let's see those maps.

plt.figure(figsize = (18, 6))



for i in range(len(cities)):

    plt.subplot(1,3,i + 1)

    city_map = map_plot(cities[i][0])

    city_map.scatter(station_df['station_longitude'], station_df['station_latitude'], 

              latlon = True, color = 'b', zorder = 10, s = 15)

    plt.title(cities[i][1])

# So I can do a heat map, I need to get a count of when the station is used. I will use the start

# as the popularity as I think that will be the most accurate.

station_count = []



for row in station_df.values:

    station_count.append(len(bike_df[bike_df['start_station_id'] == row[0]].index))



station_df['count'] = station_count

station_df.head()
plt.figure(figsize = (8,8))

bay_map = map_plot(bay_coord, res = 'h')

bay_map.scatter(station_df['station_longitude'], station_df['station_latitude'], c = station_df['count'], 

          latlon = True, zorder = 10, s = 3, cmap = 'viridis_r')

plt.title('Station Popularity')

plt.colorbar(label = 'Total Station Usage', shrink = 0.8)

plt.savefig('bay_stations.png');
fig = plt.figure(figsize = (18,6))



for i in range(3):

    ax = fig.add_subplot(1, 3, i + 1)

    city_map = map_plot(cities[i][0])

    city_map.scatter(station_df['station_longitude'], station_df['station_latitude'], c = station_df['count'], 

              latlon = True, zorder = 10, s = 15, cmap = 'viridis_r')

    plt.title(cities[i][1])

plt.savefig('city_stations.png');
plt.figure(figsize = (8,8))

bay_map = map_plot(bay_coord, res = 'h')

bay_map.hexbin(bike_df[bike_df['end_station_id'] == -1]['end_station_longitude'],

              bike_df[bike_df['end_station_id'] == -1]['end_station_latitude'], 

               cmap = 'cool', mincnt = 1, bins = 'log')

plt.title('Non-Station Drop-offs')

cb = plt.colorbar(label = 'Number in Region', shrink = 0.8)

cb.set_ticks([1, 10, 100, 1000, 10000])

cb.set_ticklabels(['1', '10', '100', '1,000', '10,000'])

plt.savefig('bay_nonstations.png');
fig = plt.figure(figsize = (18,6))



for i in range(3):

    ax = fig.add_subplot(1, 3, i + 1)

    city_map = map_plot(cities[i][0])

    city_map.hexbin(bike_df[bike_df['end_station_id'] == -1]['end_station_longitude'],

                  bike_df[bike_df['end_station_id'] == -1]['end_station_latitude'], 

                   cmap = 'cool', mincnt = 1, bins = 'log')

    plt.title(cities[i][1])

plt.savefig('city_nonstations.png');
def hourly_plot (times):

    hour_count = times.resample('H').count()

    hour_of_day_mean = hour_count.groupby(hour_count.index.time).mean()



    p = plt.bar(['12am', '1am', '2am', '3am', '4am', '5am', '6am', '7am',

            '8am', '9am', '10am', '11am', '12pm', '1pm', '2pm', '3pm',

            '4pm', '5pm', '6pm', '7pm', '8pm', '9pm', '10pm', '11pm'], hour_of_day_mean)

    px = plt.xlabel('Hour')

    py = plt.ylabel('Mean')

    return p, px, py

    

start_times_weekday = pd.Series(data = 1, index = bike_df[bike_df['start_time'].dt.weekday < 5]['start_time'])

start_times_weekend = pd.Series(data = 1, index = bike_df[bike_df['start_time'].dt.weekday > 4]['start_time'])



plt.figure(figsize = (14,8))

plt.suptitle('Mean Rides by Hour of the Day')



plt.subplot(2,1,1)

plot1, x1, y1 = hourly_plot(start_times_weekday)

plt.title('Weekdays')

plt.ylim(0, 820)



plt.subplot(2,1,2)

plot2, x2, y2 = hourly_plot(start_times_weekend)

plt.title('Weekends')

plt.ylim(0, 820)



plt.savefig('hourly_counts.png');
sns.countplot(x = bike_df['user_type'], hue = (bike_df['start_station_id'] == -1))
start_times = pd.Series(data = 1, index = bike_df['start_time'])

subs_times = pd.Series(data = 1, index = bike_df[bike_df['user_type'] == 'Subscriber']['start_time'])

cust_times = pd.Series(data = 1, index = bike_df[bike_df['user_type'] == 'Customer']['start_time'])

plt.figure(figsize = (6,5))

plt.plot(start_times.resample('W').count(), '-', label = 'All Users')

plt.plot(subs_times.resample('W').count(), '-', label = 'Subscribers')

plt.plot(cust_times.resample('W').count(), '-', label = 'Customers')

plt.title('Weekly Bike Usage')

plt.ylim(0,120000)

plt.ylabel('Count')

plt.xlabel('Time')

plt.legend()

plt.savefig('usage.png');
start_times = pd.Series(data = bike_df['duration_sec'])

start_times.index = bike_df['start_time']

day_sum = start_times.resample('D').sum()

day_count = start_times.resample('D').count()

day_of_week_mean = day_sum.groupby(day_sum.index.dayofweek).sum()/day_count.groupby(day_count.index.dayofweek).sum()

plt.bar(day_of_week_mean.index, day_of_week_mean)

plt.xticks(ticks = [0, 1, 2, 3, 4, 5, 6], labels = ['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun'])

plt.yticks(ticks = [0, 240, 480, 720, 960, 1200], labels = ['0', '4', '8', '12', '16', '20'])

plt.title('Mean Ride Length by Day of the Week')

plt.xlabel('Day')

plt.ylabel('Duration in Minutes');

#start_times
hour_sum = start_times.resample('H').sum()

hour_count = start_times.resample('H').count()

hour_of_day_mean = hour_sum.groupby(hour_count.index.time).sum()/hour_count.groupby(hour_count.index.time).sum()



plt.figure(figsize = (14,4))

plt.bar(['12am', '1am', '2am', '3am', '4am', '5am', '6am', '7am',

        '8am', '9am', '10am', '11am', '12pm', '1pm', '2pm', '3pm',

        '4pm', '5pm', '6pm', '7pm', '8pm', '9pm', '10pm', '11pm'], hour_of_day_mean)

plt.yticks(ticks = [0, 240, 480, 720, 960, 1200], labels = ['0', '4', '8', '12', '16', '20'])

plt.title('Mean Ride Length by Hour of the Day')

plt.xlabel('Hour')

plt.ylabel('Duration in Minutes');
# Get columns for these counts

time_intervals = ['12a-3a', '3a-6a', '6a-9a', '9a-12p', '12p-3p', '3p-6p', '6p-9p', '9p-12a']

for interval in time_intervals:

    station_df[interval] = 0

    

station_df.head()
station_times = bike_df.loc[:,['start_time','start_station_id']]

hour = station_times['start_time'].dt.hour

station_times['hour'] = hour



station_times.head()
for id in station_df['station_id']:

    for i in range(8):

        count = station_times[(station_times['start_station_id'] == id) & (station_times['hour'] >= (3*i))

                             & (station_times['hour'] < (3*i +2))]['hour'].count()

        station_df.at[station_df[station_df['station_id'] == id].index[0], time_intervals[i]] = count

        

station_df.head()
for i in time_intervals:

    plt.figure(figsize = (8,8))

    bay_map = map_plot(bay_coord, res = 'l')

    bay_map.scatter(station_df['station_longitude'], station_df['station_latitude'], c = station_df[i], 

              latlon = True, zorder = 10, s = 3, cmap = 'viridis_r')

    plt.title('Station Usage as ' + i)

    plt.colorbar(label = 'Total Station Usage', shrink = 0.8)

    plt.show()
for inter in time_intervals:

    fig = plt.figure(figsize = (18,6))

    plt.suptitle('Station Usage at ' + inter)



    for i in range(3):

        ax = fig.add_subplot(1, 3, i + 1)

        city_map = map_plot(cities[i][0], res = 'i')

        city_map.scatter(station_df['station_longitude'], station_df['station_latitude'], c = station_df[inter], 

                  latlon = True, zorder = 10, s = 15, cmap = 'viridis_r')

    plt.show()
#so I have a value for everything I'm changing zero to zeros in the log column, this will be slightly off,

#but really not enough to show a visual difference.

def log_and_zero(count):

    if count == 0:

        return 0

    else:

        return np.log10(count)



max_time_count_log = np.log10(station_df[time_intervals].max().max())

max_time_count_log

for i in time_intervals:

    station_df[i + '_log'] = station_df[i].apply(log_and_zero)

    

station_df.head()
# Need a maximun so everythings on the same scale.

max_log = np.log10(station_df[time_intervals].max().max())

max_log


for i in range(8):

    plt.figure(figsize = (8,8))

    bay_map = map_plot(bay_coord, res = 'h')

    bay_map.scatter(station_df['station_longitude'], station_df['station_latitude'], 

                    c = station_df[time_intervals[i] + '_log'], latlon = True, 

                    zorder = 10, s = 3, cmap = 'gist_stern_r', vmax = max_log)

    plt.title('Station Usage at ' + time_intervals[i])

    cb = plt.colorbar(label = 'Total Station Usage', shrink = 0.8)

    cb.set_ticks([0, 1, 2, 3, 4])

    cb.set_ticklabels(['1', '10', '100', '1,000', '10,000'])

    plt.savefig('bay_' + time_intervals[i] + '.png')

    plt.show();
max_usage = station_df[time_intervals].max().max()

max_usage
plt.figure(figsize = (16,8))

for i in range(8):

    plt.subplot(2, 4, i+1)

    bay_map = map_plot(bay_coord, res = 'i')

    bay_map.scatter(station_df['station_longitude'], station_df['station_latitude'], 

                    c = station_df[time_intervals[i]], latlon = True, 

                    zorder = 10, s = 2, cmap = 'viridis_r', vmax = max_usage)

    plt.title('Station Usage at ' + time_intervals[i])

    plt.colorbar(label = 'Total Station Usage', shrink = 0.8)


for i in range(8):

    fig = plt.figure(figsize = (18,6))

    for j in range(3):

        ax = fig.add_subplot(1, 3, j + 1)

        city_map = map_plot(cities[j][0], res = 'f')

        city_map.scatter(station_df['station_longitude'], station_df['station_latitude'], 

                         c = station_df[time_intervals[i] + '_log'], latlon = True, 

                         zorder = 10, s = 15, cmap = 'gist_stern_r', vmax = max_log)

        plt.title(cities[j][1])

    plt.savefig('city_' + time_intervals[i] + '.png')    

    plt.show()
def dist_globe (lat1, lon1, lat2, lon2):

    r = 3963 #radius of earth in miles

    lat1r = math.radians(lat1)

    lon1r = math.radians(lon1)

    lat2r = math.radians(lat2)

    lon2r = math.radians(lon2)

    

    a = math.sin(lat1r) * math.sin(lat2r) + math.cos(lat1r) * math.cos(lat2r) * math.cos(lon2r - lon1r)

    #There are some round off errors here getting larger than 1

    if a > 1:

        a = 1

    d = r * math.acos(a)

    return d



bike_df['distance'] = bike_df.apply(lambda x: dist_globe(x['start_station_latitude'], 

                                                         x['start_station_longitude'],

                                                         x['end_station_latitude'], 

                                                         x['end_station_longitude']),

                                   axis = 1)

bike_df.head()
plt.hist(bike_df['distance'], bins = 50, range = (0, 5));

plt.title('Distance Between Start and End Locations')

plt.xlabel('Distance in Miles')

plt.ylabel('Count')
bike_df[bike_df['distance'] > 5]['distance'].count(), bike_df['distance'].max()
# Loading up the list of BART stations and the routes

path = '/kaggle/input/bikedata/BART/'



bart_df = pd.read_csv(path + 'bart.csv')

# Ordered in the same order as in the 2019 schedule

bart_yellow_df = pd.read_csv(path + 'bart_yellow.csv')

bart_green_df = pd.read_csv(path + 'bart_green.csv')

bart_red_df = pd.read_csv(path + 'bart_red.csv')

bart_orange_df = pd.read_csv(path + 'bart_orange.csv')

bart_blue_df = pd.read_csv(path + 'bart_blue.csv')



bart_df.head()
plt.figure(figsize = (8,8))

bay_map = map_plot(bay_coord, res = 'h')

bay_map.scatter(station_df['station_longitude'], station_df['station_latitude'], c = station_df['count'], 

          latlon = True, zorder = 2, s = 3, cmap = 'viridis_r')

bay_map.plot(bart_blue_df['longitude'], bart_blue_df['latitude'], '.-', label = 'Blue Line')

bay_map.plot(bart_orange_df['longitude'], bart_orange_df['latitude'], '+--', label = 'Orange Line')

bay_map.plot(bart_green_df['longitude'], bart_green_df['latitude'], 'x-.', label = 'Green Line')

bay_map.plot(bart_red_df['longitude'], bart_red_df['latitude'], '^:', label = 'Red Line')

bay_map.plot(bart_yellow_df['longitude'], bart_yellow_df['latitude'], 'vy-', label = 'Yellow Line')

plt.title('BART Lines')

plt.legend()

plt.colorbar(label = 'Total Station Usage', shrink = 0.8)

plt.savefig('bay_bart.png');
fig = plt.figure(figsize = (18,6))



for i in range(3):

    ax = fig.add_subplot(1, 3, i + 1)

    city_map = map_plot(cities[i][0])

    city_map.scatter(station_df['station_longitude'], station_df['station_latitude'], c = station_df['count'], 

              latlon = True, zorder = 10, s = 15, cmap = 'viridis_r')

    city_map.plot(bart_blue_df['longitude'], bart_blue_df['latitude'], '.-', label = 'Blue Line')

    city_map.plot(bart_orange_df['longitude'], bart_orange_df['latitude'], '+--', label = 'Orange Line')

    city_map.plot(bart_green_df['longitude'], bart_green_df['latitude'], 'x-.', label = 'Green Line')

    city_map.plot(bart_red_df['longitude'], bart_red_df['latitude'], '^:', label = 'Red Line')

    city_map.plot(bart_yellow_df['longitude'], bart_yellow_df['latitude'], 'vy-', label = 'Yellow Line')

    plt.title(cities[i][1])

plt.savefig('city_bart.png');
sf_oak_stations = station_df[station_df['station_longitude'] <= -122.1]

sf_oak_stations.shape
def to_bart(lat, lon):

    min_dist = 100 #rediculously large so it will take the first value

    for i in bart_df.index:

        bart_lat = bart_df.loc[i, 'latitude']

        bart_lon = bart_df.loc[i, 'longitude']

        min_dist = min(min_dist, dist_globe(lat, lon, bart_lat, bart_lon))

        

    return min_dist



sf_oak_stations['bart_distance'] = sf_oak_stations.apply(lambda x: to_bart(x['station_latitude'], 

                                                                           x['station_longitude']), axis = 1)

sf_oak_stations.head()
plt.hist(sf_oak_stations['bart_distance'], bins = 25)

plt.title('Station Distance to BART Station')

plt.ylabel('Count')

plt.xlabel('Distance in Miles')
plt.scatter(sf_oak_stations['bart_distance'], sf_oak_stations['count'])
plt.scatter(sf_oak_stations['bart_distance'], np.log10(sf_oak_stations['count']))

plt.title('Station Usage and Distance to BART Station')

plt.xlabel('Distance in Miles')

plt.ylabel('Number of Uses')

plt.yticks(ticks = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5], 

           labels = ['3', '10', '30', '100', '300', '1,000', '3,000', '10,000', '30,000']);

plt.savefig('bart_usage.png');