import pandas
import numpy
%matplotlib inline

from matplotlib import pyplot

from mpl_toolkits import basemap
import shapely

import shapely.wkt
data = pandas.read_csv('../input/bikes.csv')
data.head(4)
data.info()
data.describe()
data[["bike_stands", "available_bike_stands", "available_bikes"]].describe()
data = data[data["bike_stands"] >= (data["available_bike_stands"] + data["available_bikes"])]
data.describe()
data.status.value_counts()
data.banking.value_counts()
data.bonus.value_counts()
data = data.drop(["bonus", "number", "contract_name", "banking", "address"], axis=1)

data.head(3)
data["bad_stands"] = data["bike_stands"] - (data["available_bike_stands"] + data["available_bikes"])
data.describe()
data["last_update"] = pandas.to_datetime(data["last_update"], unit='ms')

data["time"] = data['last_update'].map(lambda x: x.hour + x.minute / 60)

#data["time"] = data['last_update'].map(lambda x: x.hour)

data.head(3)
data["time_cluster"] = pandas.cut(data["time"], 24 * 3)

data["time_cluster"] = pandas.Categorical(data["time_cluster"]).codes / 3.0
grouped = data[["time_cluster", "name", "bike_stands", "available_bikes", "bad_stands"]].groupby(["time_cluster", "name"]).mean()



grouped.head(20)
# Reshape the new df

pivot = pandas.pivot_table(grouped.reset_index(), index="time_cluster", columns="name")
titles = list(pivot.available_bikes.columns.values)



# Round the times to that we do not show all 20m slots.

indexes = numpy.round(pivot.index)



ax = pivot.available_bikes.plot(subplots=True,

                 grid=True,

                 rot=0,

                 xticks=numpy.round(pivot.index),

                 figsize=(15,60),

                 title=titles,

)

ax = pivot.bad_stands.plot(subplots=True,

                      ax=ax,

                      grid=True,

                      style='r',   

)

ax = pivot.bike_stands.plot(subplots=True,

                      ax=ax,

                      grid=True,

                      style='k--',   

)



for a in ax:

    a.legend(["Available bikes", "Bad stands", "Max stands"])
bike_lanes_df = pandas.read_csv("../input/bike_lanes.csv")

bike_lanes_df["wkt_wsg84"] = bike_lanes_df["wkt_wsg84"].apply(shapely.wkt.loads)



bike_lanes_df.head(3)
bike_lanes = shapely.geometry.MultiLineString(list(bike_lanes_df["wkt_wsg84"]))
# define map colors

land_color = '#f5f5f3'

water_color = '#a4bee8'

coastline_color = '#000000'

border_color = '#bbbbbb'



map_width = 10 * 1000

map_height = 7 * 1000



# plot the map

fig_width = 20

fig = pyplot.figure(figsize=(20, 20 * map_height / map_width))



ax = fig.add_subplot(111, facecolor='#ffffff')

ax.set_title("Santander Bike Stations", fontsize=16, color='#333333')



lat = 43.47

lon = -3.82



m = basemap.Basemap(

            projection="tmerc",

            lon_0=lon, 

            lat_0=lat,

            width=map_width, 

            height=map_height,

            resolution='h',

            area_thresh=0.1

)



m.drawmapboundary(fill_color=water_color)

m.drawcoastlines(color=coastline_color)

m.drawcountries(color=border_color)

m.fillcontinents(color=land_color, lake_color=water_color)

m.drawstates(color=border_color)



means = data.groupby("name").mean()



m.scatter(means.lng.values.ravel(), 

          means.lat.values.ravel(),

          latlon=True,

          alpha=0.8,

          s=means["bike_stands"] * 5,

          label="bike stands",

          c=means["available_bikes"].astype(float),

          lw=.25,

          cmap=pyplot.get_cmap("jet"),

          zorder=3

)                



# This should work, but I do not know why it does not

#ax.add_collection(bike_lanes)

for l in bike_lanes:

    m.plot(*l.xy, latlon=True, color="grey", alpha=0.5)



c = pyplot.colorbar(orientation='vertical', shrink=0.5)

c.set_label("Available Bikes")
import cartopy.crs 

from cartopy.io import img_tiles
osm_tiles = img_tiles.OSM()
pyplot.figure(figsize=(20, 20))



# Use the tile's projection for the underlying map.

ax = pyplot.axes(projection=osm_tiles.crs)



ax.set_extent([-3.892, -3.762, 43.438, 43.495],

              cartopy.crs.PlateCarree())



# Add the tiles at zoom level 13.

ax.add_image(osm_tiles, 14)



cb = ax.scatter(means.lng.values.ravel(), 

           means.lat.values.ravel(),

           alpha=0.9,

           s=means["bike_stands"] * 5,

           label="bike stands",

           c=means["available_bikes"].astype(float),

           lw=.25,

           cmap=pyplot.get_cmap("jet"),

           zorder=3,

           transform=cartopy.crs.PlateCarree(),

)                



c = pyplot.colorbar(cb, orientation='horizontal', pad=0.04)

c.set_label("Available Bikes")



# This should work, but I do not know why it does not

#ax.add_collection(bike_lanes)

for l in bike_lanes:

    ax.plot(*l.xy, lw=2, color="red", alpha=1, transform=cartopy.crs.PlateCarree())



pyplot.show()