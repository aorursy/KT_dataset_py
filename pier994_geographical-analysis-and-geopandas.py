import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

import time
df = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
df.head()
df.info()
print("Number of houses with non positive price (data quality issue): ", (df.price <= 0).sum())
df = df[df.price > 0]
df.nunique()
print("Number of houses with a non unique position: ", df[df.duplicated(["latitude", "longitude"], keep = False)].shape[0])

print("Number of unique positions within this segment: ", df[df.duplicated(["latitude", "longitude"], keep = False)].drop_duplicates(["latitude", "longitude"]).shape[0])
df.room_type.value_counts()
sns.scatterplot(data = df,

                x = "longitude",

                y = "latitude",

                size = .5,

                legend = False)



plt.axis("off")



plt.show()
import  geopandas as gpd



ny = gpd.read_file("../input/neighborhoods-in-new-york/ZillowNeighborhoods-NY.shp")



ny = ny[ny.City == "New York"]



ny.head()
ny["Area"] = ny.geometry.area
gdf = gpd.GeoDataFrame(

    df, geometry=gpd.points_from_xy(df.longitude, df.latitude),

    crs={'init':'epsg:4326'})
gdf.head()
fig, ax = plt.subplots(1, 2, figsize = (15,10))

gdf.plot("neighbourhood_group", markersize = .7, ax = ax[0], legend = True)

ny.plot(column = "County", legend = True, alpha = .4, ax = ax[1])



plt.plot()
ny["County"].replace({"New York": "Manhattan",

                     "Kings": "Brooklyn",

                     "Richmond": "Staten Island"

                    }, inplace = True)
fig, ax = plt.subplots(figsize = (15,10))



ny.plot(column = "County", legend = True, alpha = .4, ax = ax)

gdf.plot(markersize = .7, ax = ax)



plt.axis("off")



plt.plot()
gdf = gpd.sjoin(gdf[["neighbourhood_group", "neighbourhood", "price", "minimum_nights", "geometry"]], 

                ny[["County", "Name", "Area", "geometry"]])



gdf.head()
d = gdf.groupby(["Name"]).size().reset_index()

d.columns = ["Name", "N_Houses"]



d.head()
d = ny.merge(d,

         how = "left",

         on = "Name"

        )



d["Density"] = d["N_Houses"]/d["Area"]



d.head()
fig, ax = plt.subplots(figsize=(10,8))

d.plot(ax = ax,

        cmap = sns.cubehelix_palette(start = 2.5, 

                                     rot = .1, 

                                     gamma = .7,

                                     hue = .7, 

                                     light = .8,

                                     dark = 0, 

                                     as_cmap = True

                                    ),

        column = "Density",

       edgecolor = "#ffffff"

       )



plt.axis("off")



plt.show()
import matplotlib.cm as cm

from matplotlib.colors import Normalize 
df.head()
df.price.describe()
df.price.quantile(np.arange(0., 1.01, .05))
norm = Normalize(vmin=0,

                 vmax=250,

                 clip=True

                 )
fig, ax = plt.subplots(1, figsize=(15,10))



ny.plot(ax = ax, edgecolor = "#a6a6a6", color = "white")



sns.scatterplot(data = df,

                y = "latitude",

                x = "longitude", 

                hue = "price", 

                palette = "magma",

                size = .3,

                hue_norm = norm,

                legend = False, ax = ax

               )



cbar = cm.ScalarMappable(norm=norm, cmap='magma')

cbar = fig.colorbar(cbar, 

                   ticks = [0,

                            50,

                            100,

                            150,

                            200,

                            250

                           ], 

                    ax = ax)



plt.axis("off")



cbar.ax.set_yticklabels(["€0", "€50", "€100", "€150", "€200", "more than €250"], fontsize = 15)



ax.set_title("Map of airbnb house prices in NY", fontsize = 15)



plt.show()







# plt.xlim(-74.5, -73)

# plt.ylim(40.5,41)
df.room_type.unique()
np.seterr(under='ignore')

sns.set(font_scale=2)

fg = sns.FacetGrid(data = df[df.price < 500], col = "neighbourhood_group", col_wrap = 3, height = 9)

fg.map(sns.violinplot, "room_type", "price")

plt.show()
fig, ax = plt.subplots(2, 2, figsize=(20,20))



for a in ax.reshape(-1)[:3]:

    a.axis("off")

    ny.plot(ax = a, edgecolor = "#a6a6a6", color = "white")

    

ax[1,1].axis("off")



sns.scatterplot(data = df[df.room_type == "Shared room"],

                y = "latitude",

                x = "longitude", 

                hue = "price", 

                palette = "magma",

                size = .3,

                hue_norm = norm,

                legend = False, ax = ax[0,0]

               )



sns.scatterplot(data = df[df.room_type == "Private room"],

                y = "latitude",

                x = "longitude", 

                hue = "price", 

                palette = "magma",

                size = .3,

                hue_norm = norm,

                legend = False, ax = ax[0,1]

               )



sns.scatterplot(data = df[df.room_type == "Entire home/apt"],

                y = "latitude",

                x = "longitude", 

                hue = "price", 

                palette = "magma",

                size = .3,

                hue_norm = norm,

                legend = False, ax = ax[1,0]

               )





cbar = cm.ScalarMappable(norm=norm, cmap='magma')

cbar = fig.colorbar(cbar, 

                   ticks = [0,

                            50,

                            100,

                            150,

                            200,

                            250

                           ], 

                    ax = ax[1, 1])







cbar.ax.set_yticklabels(["€0", "€50", "€100", "€150", "€200", "more than €250"], fontsize = 15)



ax[0, 0].set_title("Prices of Shared Rooms", fontsize = 15)

ax[0, 1].set_title("Prices of Private Rooms", fontsize = 15)

ax[1, 0].set_title("Prices of Entire homes or apartments", fontsize = 15)



plt.show()
subway = gpd.read_file("../input/ny-geodata/geo_export_3aa73ff9-4ce9-4951-8d6c-2fd706580916.shp", crs={"init": "epsg:4326"})
subway.head()
gdf = gpd.GeoDataFrame(

    df, geometry=gpd.points_from_xy(df.longitude, df.latitude),

    crs={'init':'epsg:4326'})



# Creation of the projected data frames

subway_xy = subway.to_crs(epsg = 3763)

gdf_xy = gdf.to_crs(epsg=3763)



# Creation of three data frames representing the areas near the subway stations

subway_xy200 = gpd.GeoDataFrame({"200mt": [1]*473, "geometry": subway_xy.buffer(200)})

subway_xy500 = gpd.GeoDataFrame({"500mt": [1]*473, "geometry": subway_xy.buffer(500)})

subway_xy1 = gpd.GeoDataFrame({"1km": [1]*473, "geometry": subway_xy.buffer(1000)})
# House data frame is merged with the data of subway proximity. Notice that it is possible that an house might be 

# within 1000 (or 500 or 200) metres away from multiple subway stations. Hence duplicated id must be removed



gdf_xy = gpd.sjoin(gdf_xy, subway_xy200, how = "left")

gdf_xy.drop("index_right", axis = 1, inplace = True)



gdf_xy = gpd.sjoin(gdf_xy, subway_xy500, how = "left")

gdf_xy.drop("index_right", axis = 1, inplace = True)



gdf_xy = gpd.sjoin(gdf_xy, subway_xy1, how = "left")

gdf_xy.drop("index_right", axis = 1, inplace = True)



gdf_xy[["200mt", "500mt", "1km"]] = gdf_xy[["200mt", "500mt", "1km"]].fillna(0)



gdf_xy = gdf_xy.groupby("id").agg({"200mt": "max",

                          "500mt": "max",

                          "1km": "max",

                        })



# These new features are merged in the original geo data frame

gdf = gdf.merge(gdf_xy, how = "left", on = "id")
gdf_xy.head()
display(gdf["200mt"].value_counts())

display(gdf["500mt"].value_counts())

display(gdf["1km"].value_counts())
# Some data quality check (if the nearest subway station is within 200 meters, it must be also within 1000 meters)

display(gdf.loc[gdf["200mt"] == 1, "500mt"].value_counts())

display(gdf.loc[gdf["200mt"] == 1, "1km"].value_counts())

display(gdf.loc[gdf["500mt"] == 1, "1km"].value_counts())
gdf["SubwayLoc"] = "No Near Subways"

gdf.loc[gdf["1km"] == 1, "SubwayLoc"] = "Subway within 1km"

gdf.loc[gdf["500mt"] == 1, "SubwayLoc"] = "Subway within 500mt"

gdf.loc[gdf["200mt"] == 1, "SubwayLoc"] = "Subway within 200mt"
gdf.SubwayLoc.value_counts()
gdf["P"] = gdf["price"]



gdf.loc[gdf["P"] > 500, "P"] = 500
np.seterr(under='ignore')

sns.set(font_scale=2)

fg = sns.FacetGrid(data = gdf, col = "room_type", col_wrap = 3, height = 9)

fg.map(sns.violinplot, "SubwayLoc", "P")

fg.set_xticklabels(rotation=30)

plt.show()
np.seterr(under='ignore')

sns.set(font_scale=2)

fg = sns.FacetGrid(data = gdf[gdf["neighbourhood_group"] == "Manhattan"], col = "room_type", col_wrap = 3, height = 9)

fg.map(sns.violinplot, "SubwayLoc", "P")

fg.set_xticklabels(rotation=30)

plt.show()