import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
df=pd.read_csv('../input/metro-bike-share-trip-data.csv')
df.head()
df_R=df[df['Trip Route Category']=='Round Trip']
df_O=df[df['Trip Route Category']=='One Way']
df_R=df_R.drop(['Ending Station Latitude','Ending Station Longitude','Trip Route Category'],axis=1)
df_R=df_R[(df_R['Starting Station Latitude']>33.5)&(df_R['Starting Station Latitude']<35)]
df_R=df_R[(df_R['Starting Station Longitude']>-119)&(df_R['Starting Station Longitude']<-117)]
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(15,20))
ax=sns.scatterplot(x="Starting Station Longitude", y="Starting Station Latitude", hue="Passholder Type",data=df_R)
df_O=df_O[(df_O['Starting Station Latitude']>33.5)&(df_O['Starting Station Latitude']<35)]
df_O=df_O[(df_O['Starting Station Longitude']>-119)&(df_O['Starting Station Longitude']<-117)]

df_O=df_O[(df_O['Ending Station Latitude']>33.5)&(df_O['Ending Station Latitude']<35)]
df_O=df_O[(df_O['Ending Station Longitude']>-119)&(df_O['Ending Station Longitude']<-117)]
plt.figure(figsize=(15,20))
ax=sns.scatterplot(x="Starting Station Longitude", y="Starting Station Latitude", hue="Passholder Type",data=df_O, color='red')
ax=sns.scatterplot(x="Ending Station Longitude", y="Ending Station Latitude", hue="Passholder Type",data=df_O, color='green')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly import tools
from subprocess import check_output
mapbox_access_token='pk.eyJ1IjoiYW1tb24xIiwiYSI6ImNqbGhtdDNtNzFjNzQzd3J2aDFndDNmbmgifQ.-dt3pKGSvkBaSQ17qXVq3A'
data = [
    go.Scattermapbox(
        lat=df_O['Starting Station Latitude'],
        lon=df_O['Starting Station Longitude'],
        mode='markers',
        marker=dict(
            size=5,
            color='red',
            opacity=0.3
        )),
    ]
layout = go.Layout(
    autosize=True,
    hovermode='closest',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=34.050777,
            lon=-118.225554
        ),
        pitch=0,
        zoom=12.48,
        style='light'
    ),
)

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='Crime')
df_R.head()
df_R['hour_s']=df_R['Start Time'].str[11:13]
df_R.hour_s = pd.to_numeric(df_R.hour_s)
df_R['hour_e']=df_R['End Time'].str[11:13]
df_R.hour_e = pd.to_numeric(df_R.hour_e)

df_R['min_s']=df_R['Start Time'].str[14:16]
df_R.min_s = pd.to_numeric(df_R.min_s)
df_R['min_e']=df_R['End Time'].str[14:16]
df_R.min_e = pd.to_numeric(df_R.min_e)
df_R['time']=df_R['hour_e']-df_R['hour_s']+(1/60)*(df_R['min_e']-df_R['min_s'])
df_R.loc[df_R.time<0,['time']]=df_R['time']+24
df_R.head()
import seaborn as sns
sns.scatterplot(x="Duration", y="time", data=df_R)
sns.scatterplot(x="Duration", y="Plan Duration", data=df_R)
sns.scatterplot(x="Duration", y="Starting Station ID", data=df_R)