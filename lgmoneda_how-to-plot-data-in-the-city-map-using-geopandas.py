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



city = ox.gdf_from_place('São Paulo, São Paulo, Brasil')
fig, ax = plt.subplots(figsize=(15, 15))

city.plot(ax=ax)
data = pd.read_csv("/kaggle/input/eleicoes-municipais-2016-sp/sao_paulo.csv", converters={"coordenadas": eval})
data.head()
data["geometry"] = data["coordenadas"].apply(lambda x : Point(x))
geo_df = gpd.GeoDataFrame(data, crs={"init": "epsg:4326"}, geometry=data["geometry"])
exclude_data = geo_df[geo_df["coordenadas"] >= (0, 0)]["coordenadas"].unique()

geo_df = geo_df[~geo_df["coordenadas"].isin(exclude_data)]
geo_df.head()
agg_geo_df = geo_df.groupby("endereço")["geometry"].first()
fig, ax = plt.subplots(figsize=(15, 15))

city.plot(ax=ax, color="grey", alpha=0.4)

agg_geo_df.plot(ax=ax, markersize=4, color="red", marker="o")
SP_cities = gpd.read_file("/kaggle/input/eleicoes-municipais-2016-sp/sp_municipios/35MUE250GC_SIR.shp")

SP_cities.head()
sp_city = SP_cities[SP_cities["NM_MUNICIP"] == "SÃO PAULO"]["geometry"].values[0]
agg_geo_df = geo_df.groupby("endereço", as_index=False)["geometry"].first()

agg_geo_df = agg_geo_df[agg_geo_df["geometry"].apply(lambda x: x.within(sp_city))]
fig, ax = plt.subplots(figsize=(15, 15))

city.plot(ax=ax, color="grey", alpha=0.4)

agg_geo_df.plot(ax=ax, markersize=4, color="red", marker="o")
from sklearn.preprocessing import MinMaxScaler



mm = MinMaxScaler()
candidate = "EDUARDO SUPLICY"



agg = data.groupby(["endereço", "candidato"], as_index=False).agg({"votos": "sum", 

                                                                   "coordenadas": "first",

                                                                   "zona": "first",

                                                                   "geometry": "first"})
agg = agg[agg["candidato"] == candidate] 

agg = agg[agg["geometry"].apply(lambda x: x.within(sp_city))]
geo_agg = gpd.GeoDataFrame(agg, crs={"init": "epsg:4326"}, geometry=agg["geometry"])

geo_agg["votos_size"] = mm.fit_transform(geo_agg[["votos"]])
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