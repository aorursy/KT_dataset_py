import folium

import pandas as pd

import numpy as np
path="/kaggle/input/corona-analysis/finalcopy-Copy1.csv"

df1=pd.read_csv(path)
df1
from folium import plugins



# let's start again with a clean copy of the map of San Francisco

coronamap2 = folium.Map( zoom_start = 12, tiles='Stamen Terrain')



# instantiate a mark cluster object for the incidents in the dataframe

incidents = plugins.MarkerCluster().add_to(coronamap2)



# loop through the dataframe and add each data point to the mark cluster

for lat,lng,susp,confirm,death,d,m,y  in zip(df1["Latitude"], df1["Longitude"],df1["Suspected"],df1["Confirmed"],df1["Deaths"],df1["D"],df1["M"],df1["Y"]):

    folium.Marker(

        location=[lat, lng],

        icon=None,

        popup=("Date:"+str(d)+"-"+str(m)+"-"+str(y)+'<br>'+"Suspected:"+str(susp)+'<br>'+"Confirmed:"+str(confirm)+'<br>'+"Death:"+str(death)),

    ).add_to(incidents)



# display map

coronamap2