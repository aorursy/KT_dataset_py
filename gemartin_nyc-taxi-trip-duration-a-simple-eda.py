import numpy as np

import pandas as pd

import seaborn.apionly as sns

import matplotlib.pyplot as plt

from datetime import date, datetime

from haversine import haversine



# statistics package

import statsmodels.api as sm

from statsmodels.formula.api import ols

from scipy import stats



# packages for mapping

from mpl_toolkits.basemap import Basemap



# packages for interactive graphs

from ipywidgets import widgets, interact

from IPython.display import display



%matplotlib inline
def data_distribution(data):

    """ Draws a chart showing data distribution

        by combining an histogram and a boxplot

        

    Parameters

    ----------

    data: array or series

        the data to draw the distribution for

        

    """

    

    x = np.array(data)

    

    # set the number of bins using the Rice rule

    # n_bins = twice cube root of number of observations

    n = len(x)

    n_bins = round(2 * n**(1/3))

    

    fig = plt.figure()

    

    # histogram

    ax1 = fig.add_axes([0.1, 0.3, 0.8, 0.6])

    ax1 = plt.hist(x, bins=n_bins, alpha=0.7)

    plt.grid(alpha=.5)

    

    # boxplot

    ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.2])

    ax2 = plt.boxplot(x, vert=False, widths=0.7)

    plt.grid(alpha=.5)

           

    plt.show()
def distance(lat1, lon1, lat2, lon2):

    """calculates the Manhattan distance between 2 points

        using their coordinates

    

    Parameters

    ----------

    lat1: float

        latitude of first point

        

    lon1: float

        longitude of first point

        

    lat2: float

        latitude of second point

    

    lon2: float

        longitude of second point

        

    Returns

    -------

    d: float

        The Manhattan distance between the two points in kilometers

        

    """

    

    d = haversine((lat1, lon1), (lat2, lon1)) + haversine((lat2, lon1), (lat2, lon2))

    return d
df = pd.read_csv("../input/train.csv")

print("Rows: {}".format(df.shape[0]))

print("Columns: {}".format(df.shape[1]))
df.info()
df.head()
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])
df["pickup_month"] = df["pickup_datetime"].apply(lambda x: x.month)

df["pickup_day"] = df["pickup_datetime"].apply(lambda x: x.day)

df["pickup_weekday"] = df["pickup_datetime"].apply(lambda x: x.weekday())

df["pickup_hour"] = df["pickup_datetime"].apply(lambda x: x.hour)

df["pickup_minute"] = df["pickup_datetime"].apply(lambda x: x.minute)

df["pickup_time"] = df["pickup_hour"] + (df["pickup_minute"] / 60)



df["dropoff_hour"] = df["dropoff_datetime"].apply(lambda x: x.hour)
# The distance is calculated in kilometers

df["distance"] = df.apply(lambda row: distance(row["pickup_latitude"], 

                                               row["pickup_longitude"], 

                                               row["dropoff_latitude"], 

                                               row["dropoff_longitude"]), axis=1)
# The speed is calculated in km/h

df["speed"] = df["distance"] / (df["trip_duration"] / 3600)
flags = {"N":0, "Y":1}

df["store_and_fwd_flag"] = df["store_and_fwd_flag"].map(flags)
df.info()
df["trip_duration"].describe()
df[["trip_duration", "vendor_id", "passenger_count", "store_and_fwd_flag", "distance", "speed"]][df["trip_duration"] > 36000].shape[0]
data_distribution(df["trip_duration"][df["trip_duration"] <= 3600])
plt.figure(figsize=(20,20))



# Set the limits of the map to the minimum and maximum coordinates

lat_min = df["pickup_latitude"].min() - .2

lat_max = df["pickup_latitude"].max() + .2

lon_min = df["pickup_longitude"].min() - .2

lon_max = df["pickup_longitude"].max() + .2



# Set the center of the map

cent_lat = (lat_min + lat_max) / 2

cent_lon = (lon_min + lon_max) / 2



map = Basemap(llcrnrlon=lon_min, llcrnrlat=lat_min, urcrnrlon=lon_max, urcrnrlat=lat_max,

             resolution='h', projection='tmerc', lat_0 = cent_lat, lon_0 = cent_lon)



map.drawmapboundary(fill_color='aqua')

map.fillcontinents(color='lightgray', lake_color='aqua')

map.drawcountries(linewidth=2)

map.drawstates(color='b')



long = np.array(df["pickup_longitude"])

lat = np.array(df["pickup_latitude"])



x, y = map(long, lat)

map.plot(x, y,'ro', markersize=3, alpha=1)



plt.show()
df[["id", "distance", "trip_duration", "speed"]][df["pickup_longitude"] == lon_min + .2]
df[["id", "distance", "trip_duration", "speed"]][df["pickup_latitude"] == lat_max - .2]
plt.figure(figsize=(20,20))



# Set the limits of the map to the minimum and maximum coordinates

lat_min = 40.6

lat_max = 40.9

lon_min = -74.2

lon_max = -73.7



# Set the center of the map

cent_lat = (lat_min + lat_max) / 2

cent_lon = (lon_min + lon_max) / 2



map = Basemap(llcrnrlon=lon_min, llcrnrlat=lat_min, urcrnrlon=lon_max, urcrnrlat=lat_max,

             resolution='h', projection='tmerc', lat_0 = cent_lat, lon_0 = cent_lon)



map.drawmapboundary(fill_color='aqua')

map.fillcontinents(color='lightgray', lake_color='aqua')

map.drawcountries(linewidth=2)

map.drawstates(color='b')



long = np.array(df["pickup_longitude"])

lat = np.array(df["pickup_latitude"])



x, y = map(long, lat)

map.plot(x, y,'ro', markersize=2, alpha=0.2)



plt.show()
lm = ols("trip_duration ~ pickup_latitude + pickup_longitude", data=df).fit()

print(lm.summary())
df["distance"].describe()
data_distribution(df["distance"])
data_distribution(df["distance"][df["distance"] <= 100])
lm = ols("trip_duration ~ distance", data=df).fit()

print(lm.summary())
df["speed"].describe()
data_distribution(df["speed"])
data_distribution(df["speed"] <= 50)
lm = ols("trip_duration ~ speed", data=df).fit()

print(lm.summary())
lm = ols("trip_duration ~ speed", data=df[(df["speed"] >= 11.50) & (df["speed"] <= 23.15)]).fit()

print(lm.summary())
sns.countplot(df["pickup_hour"])
g = sns.FacetGrid(df, col="pickup_weekday")

g.map(plt.hist, "pickup_hour");
fig = plt.figure(figsize=(10,10))



x = df["pickup_time"][df["speed"] < 100]

y = df["speed"][df["speed"] < 100]



plt.scatter(x=x, y=y, alpha=0.01)



plt.xlabel("Pickup time (h.m of day)")

plt.ylabel("Average speed (km/h)")

plt.grid(alpha=0.5)

plt.show()
fig, ax = plt.subplots (4, 2, figsize=(15, 15))



d = 0



days = {0:"Monday", 1:"Tuesday", 2:"Wednesday", 3:"Thursday", 4:"Friday",

        5:"Saturday", 6:"Sunday"}



for r in range(0, 4):

    for c in range(0, 2):

        if d > 6:

            ax[r, c].axis("off")

            break

        x = df["pickup_time"][(df["speed"] < 100) & (df["pickup_weekday"] == d)]

        y = df["speed"][(df["speed"] < 100) & (df["pickup_weekday"] == d)]



        ax[r, c].scatter(x=x, y=y, alpha=0.01)

        ax[r, c].set_title("{}".format(days[d]))

        ax[r, c].axhline(40, linewidth=1, color='r', linestyle="--", alpha=.5)

        ax[r, c].grid(alpha=0.5)

        d += 1



fig.suptitle("Observed average speeds depending on day of week and time of day")

plt.show()
# Set the limits of the map to the minimum and maximum coordinates

lat_min = 40.6

lat_max = 40.9

lon_min = -74.05

lon_max = -73.75



# Set the center of the map

cent_lat = (lat_min + lat_max) / 2

cent_lon = (lon_min + lon_max) / 2



columns = ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude", "pickup_hour"]

sample = df[columns][(df["pickup_latitude"] >= lat_min) & \

                      (df["pickup_latitude"] <= lat_max) & \

                      (df["pickup_longitude"] >= lon_min) & \

                      (df["pickup_longitude"] <= lon_max) & \

                      (df["speed"] >= 10) & \

                      (df["speed"] <= 60)]





def draw_map(hour):

    fig = plt.figure(figsize=(20, 20))

    

    # plot pickups

    ax = fig.add_subplot(121)

    ax.set_title("Pickups")

    

    # map definition

    map = Basemap(llcrnrlon=lon_min, llcrnrlat=lat_min, urcrnrlon=lon_max, urcrnrlat=lat_max,

                resolution='h', projection='tmerc', lat_0 = cent_lat, lon_0 = cent_lon)



    map.drawmapboundary(fill_color='aqua')

    map.fillcontinents(color='lightgray', lake_color='aqua')

    

    lon = np.array(sample["pickup_longitude"][sample["pickup_hour"] == hour])

    lat = np.array(sample["pickup_latitude"][sample["pickup_hour"] == hour])

    x, y = map(lon, lat)

    map.plot(x, y,'bo', markersize=1, alpha=0.3)



    # plot dropoffs

    ax = fig.add_subplot(122)

    ax.set_title("Dropoffs")

    

    # map definition

    map = Basemap(llcrnrlon=lon_min, llcrnrlat=lat_min, urcrnrlon=lon_max, urcrnrlat=lat_max,

                resolution='h', projection='tmerc', lat_0 = cent_lat, lon_0 = cent_lon)



    map.drawmapboundary(fill_color='aqua')

    map.fillcontinents(color='lightgray', lake_color='aqua')

    

    lon = np.array(sample["dropoff_longitude"][sample["pickup_hour"] == hour])

    lat = np.array(sample["dropoff_latitude"][sample["pickup_hour"] == hour])

    x, y = map(lon, lat)

    map.plot(x, y,'ro', markersize=1, alpha=0.3)



    plt.show()



interact(draw_map, hour=widgets.IntSlider(min=0,max=23,step=1,value=12))
df["passenger_count"].describe()
sns.countplot(df["passenger_count"])
lm = ols("trip_duration ~ passenger_count", data=df).fit()

print(lm.summary())
sns.countplot(df["vendor_id"])
lm = ols("trip_duration ~ vendor_id", data=df).fit()

print(lm.summary())
# pickup per hour per vendor

vendor1 = df["pickup_hour"][df["vendor_id"] == 1].value_counts()

vendor2 = df["pickup_hour"][df["vendor_id"] == 2].value_counts()

fig = plt.figure()

plt.scatter(x=vendor1.index, y = vendor1, color='r', alpha=.5)

plt.scatter(x=vendor2.index, y = vendor2, color='b', alpha =.5)

plt.title("Total number of pickups per hour")

plt.xlabel("hour of the day")

plt.ylabel("Number of pickups")

plt.show()
print("Store and forward flag = 0")

data_distribution(df["trip_duration"][df["store_and_fwd_flag"] == 0])



print("Store and forward flag = 1")

data_distribution(df["trip_duration"][df["store_and_fwd_flag"] == 1])
print("Store and forward flag = 0")

data_distribution(df["speed"][df["store_and_fwd_flag"] == 0])



print("Store and forward flag = 1")

data_distribution(df["speed"][df["store_and_fwd_flag"] == 1])
df["speed"][df["store_and_fwd_flag"] == 1].describe()
corr = df.corr().mul(100).astype(int)

cg = sns.clustermap(data=corr, annot=True, fmt='d')

plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

plt.show()
lm = ols("passenger_count ~ vendor_id", data=df).fit()

print(lm.summary())