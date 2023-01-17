import os

import matplotlib.pyplot as plt

from matplotlib import style

style.use("fivethirtyeight")

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

import pandas as pd

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
dict="/kaggle/input/covid-19 kaggle/corona virus report/covid_19_clean_complete.csv"
coviddataset=pd.read_csv("/kaggle/input/covid-19 kaggle/corona virus report/covid_19_clean_complete.csv")
coviddataset.head()
coviddataset.isnull().sum()
coviddataset["day"]=coviddataset["Date"].str.split('/').str[1].astype(int)

coviddataset["month"]=coviddataset["Date"].str.split('/').str[0].astype(int)

coviddataset["year"]=coviddataset["Date"].str.split('/').str[2].astype(int)

coviddataset.head()
plt.figure(figsize=(10,6))

coviddataset.groupby("month").mean()["Confirmed"].plot()

plt.xlabel("month")

plt.ylabel("cases conformed")

plt.title("No of positive cases conformed")
plt.figure(figsize=(10,6))

coviddataset.groupby("month").mean()["Deaths"].plot()

plt.xlabel("month")

plt.ylabel("Deaths conformed")

plt.title("No of Deaths")
plt.figure(figsize=(10,6))

coviddataset.groupby("month").mean()["Recovered"].plot()

plt.xlabel("month")

plt.ylabel("Recovered cases")

plt.title("Recovered")
plt.figure(figsize=(10,6))

coviddataset.groupby("month")["Confirmed"].plot()

plt.title("no of cases conformed")
plt.figure(figsize=(10,6))

coviddataset.groupby("month")["Deaths"].plot()

plt.title("no of deaths conformed")
plt.figure(figsize=(10,6))

coviddataset.groupby("month")["Recovered"].plot()

plt.title("no of Recovered")
ax=plt.figure(figsize=(19,12.5))

ax.add_subplot(121)

sns.lineplot(x="month",y="Confirmed",data=coviddataset,color="r")

ax.add_subplot(122)

sns.lineplot(x="month",y="Deaths",data=coviddataset,color="black")
fig=plt.figure(figsize=(19,10))

ax=fig.add_subplot(121,projection="3d")

ax.scatter(coviddataset["Confirmed"],coviddataset["Recovered"],coviddataset["Deaths"],color="r")

ax.set(xlabel='\nConfirmed',ylabel='\nRecovered',zlabel='\nDeaths')
import plotly.graph_objects as go
x=coviddataset.sort_values("Confirmed",ascending=False)["Confirmed"][0:30]

fig=go.Figure(

                    data=[go.Bar(y=list(x))],

                    layout_title_text="Highest no of cases per day"

                   )

fig.data[0].marker.line.width=2

fig.data[0].marker.line.color="black"

fig.show()
y=coviddataset.sort_values("Deaths",ascending=False)["Deaths"][0:30]

fig=go.Figure(

                    data=[go.Bar(y=list(y))],

                    layout_title_text="Highest no of Deaths per day"

                   )

fig.data[0].marker.line.width=2

fig.data[0].marker.line.color="black"

fig.show()
from plotly.subplots import make_subplots
dataset=coviddataset[["Recovered","Deaths","Confirmed"]]
#make figure with subplots

fig=make_subplots(rows=1,cols=1,specs=[[{"type":"surface"}]])

#adding surface

fig.add_surface(z=dataset)

fig.update_layout(

                   showlegend=False,

                   height=800,

                   width=800,

                   title_text="3D model"

                 )

fig.show()
import plotly.express as px
df=coviddataset

fig=px.scatter_3d(df,x="Confirmed",y="Recovered",z="Deaths")

fig.show()
#px.set_mapbox_access_token(open(".mapbox_token").read())

fig=px.scatter_mapbox(df,lat="Lat",lon="Long",size="Confirmed",color="Deaths",size_max=40,zoom=10,

                     color_continuous_scale=px.colors.cyclical.IceFire)

fig.update_layout(

                  mapbox_style="white-bg",

                  mapbox_layers=[{

                                   "below": 'traces',

                                   "sourcetype": "raster",

                                   "source": [

                                    "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"

                                            ]

                                }]

                 )

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
plt.figure(figsize=(10,6))

sns.scatterplot(x="Confirmed",y="Long",data=coviddataset,color="red")
plt.figure(figsize=(10,6))

sns.scatterplot(x="Confirmed",y="Lat",data=coviddataset,color="green")
plt.figure(figsize=(10,6))

sns.scatterplot(x="Long",y="Lat",hue="Confirmed",data=coviddataset)
import folium

import webbrowser

from folium.plugins import HeatMap
latitude=30.5

longitude=114.3

dup=coviddataset.copy()

dup["count"]=1

dup.head()

p="orgin"

def worldmap(location=[latitude,longitude],zoom=9):

    map=folium.Map(location=location,control_state=True,zoom_start=zoom)

    return map

fmap=worldmap()

folium.TileLayer("cartodbpositron").add_to(fmap)

sg=folium.FeatureGroup(name="target").add_to(fmap)

folium.Marker([latitude,longitude],

              popup=p,

              icon=folium.Icon(color="red")).add_to(sg)

HeatMap(data=dup[["Lat","Long","count"]].groupby(["Lat","Long"]).sum().reset_index().values.tolist(),

       radius=8,max_zoom=13,name='Heat Map').add_to(fmap)

folium.LayerControl(collapsed=False).add_to(fmap)

fmap