# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import osmnx as ox

import pandas as pd

import matplotlib.pyplot as plt

import geopandas as gpd



from shapely.geometry import Point

from sklearn.preprocessing import MinMaxScaler



city = ox.gdf_from_place('São Paulo, São Paulo, Brasil')

mm = MinMaxScaler()
data = pd.read_csv("/kaggle/input/eleicoes-municipais-2016-sp/sao_paulo.csv", converters={"coordenadas": eval})

data["geometry"] = data["coordenadas"].apply(lambda x : Point(x))
SP_cities = gpd.read_file("/kaggle/input/eleicoes-municipais-2016-sp/sp_municipios/35MUE250GC_SIR.shp")

SP_cities.head()
sp_city = SP_cities[SP_cities["NM_MUNICIP"] == "SÃO PAULO"]["geometry"].values[0]
candidate = "EDUARDO SUPLICY"



agg = data.groupby(["endereço", "candidato"], as_index=False).agg({"votos": "sum", 

                                                                   "coordenadas": "first",

                                                                   "zona": "first",

                                                                   "geometry": "first"})



agg = agg[agg["candidato"] == candidate] 

agg = agg[agg["geometry"].apply(lambda x: x.within(sp_city))]



print("{} has {} votes".format(candidate, agg["votos"].sum()))



geo_agg = gpd.GeoDataFrame(agg, crs={"init": "epsg:4326"}, geometry=agg["geometry"])

geo_agg["votos_size"] = mm.fit_transform(geo_agg[["votos"]])
geo_agg.shape
max_amount = float(geo_agg["votos"].max())



fig, ax = plt.subplots(figsize=(15, 15))

city.plot(ax=ax, color="grey", alpha=0.4)





### We plot all locations in red to represent without votes. If there are votes they, they will be covered

geo_agg.plot(ax=ax, markersize=5.5, color="red", marker="o", label="Seções sem votos")





geo_agg.plot(column="votos", 

             ax=ax, 

             markersize=100 * geo_agg["votos_size"], 

             marker="o", 

             vmax=max_amount,

             cmap="Greens", 

             legend=True,

             label="Seções com votos")



plt.xticks([], [])

plt.yticks([], [])

plt.title("Votos de " + candidate + " por região")

plt.legend()

plt.show()
sp_zones = gpd.read_file("/kaggle/input/eleicoes-municipais-2016-sp/SP_ZONAS_janeiro_2018/ZONAS_FINAL.shp")

sp_zones.head()
sp_zones.shape
agg_with_sectors = gpd.sjoin(sp_zones, geo_agg, how="inner", op='intersects')
agg_with_sectors.columns
agg_with_sectors = agg_with_sectors.groupby("FIRST_NOME", as_index=False).agg({"votos": "sum", "geometry": "first"})

agg_with_sectors = gpd.GeoDataFrame(agg_with_sectors, crs={"init": "epsg:4326"}, geometry=agg_with_sectors["geometry"])
agg_with_sectors.head()
agg_with_sectors.shape
agg_with_sectors["votos"].sum()
variable = "votos"



fig, ax = plt.subplots(figsize=(15, 15))

city.plot(ax=ax, color="#ece5e5")



vmax = agg_with_sectors[variable].quantile(0.99)

vmin = agg_with_sectors[variable].quantile(0.01)



ax = agg_with_sectors.plot(column=variable, 

         cmap="Greens", 

         legend=True,

         ax=ax,

         k=10,

         vmax=vmax,

         vmin=vmin

         )





plt.xticks([], [])

plt.yticks([], [])

plt.title("Votos de " + candidate + " por zona eleitoral")

plt.legend()

plt.show()
variable = "votos"

agg_with_sectors["name"] = agg_with_sectors["FIRST_NOME"].apply(lambda x: x.split(" - ")[1])



fig, ax = plt.subplots(figsize=(35, 35))

city.plot(ax=ax, color="#ece5e5")



vmax = agg_with_sectors[variable].quantile(0.99)

vmin = agg_with_sectors[variable].quantile(0.01)



ax = agg_with_sectors.plot(column=variable, 

         cmap="Greens", 

         legend=True,

         ax=ax,

         k=10,

         vmax=vmax,

         vmin=vmin

         )



agg_with_sectors.apply(lambda x: ax.annotate(s=x["name"], xy=x.geometry.centroid.coords[0], ha='center', color="blue"), axis=1);

plt.xticks([], [])

plt.yticks([], [])

plt.title("Votos de " + candidate + " por zona eleitoral")

plt.legend()

plt.show()