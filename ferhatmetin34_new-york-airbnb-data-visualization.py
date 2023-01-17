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
import seaborn as sns

import matplotlib.pyplot as plt

import folium



from folium import plugins

import branca.colormap as cm

import missingno as msno

import plotly.express as px

import plotly.offline as py

import plotly.graph_objs as go

!pip install dython

from dython import nominal
sns.set(style="darkgrid",palette="muted")

pd.set_option('display.precision',7)

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows',None)

pd.options.display.float_format = '{:.7f}'.format
data=pd.read_csv("/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")

df=data.copy()

df.head()
print("Row Count : {} \nColumn Count : {}".format(df.shape[0],df.shape[1]))


#fig = px.scatter_geo(df,lat="latitude",lon="longitude",projection="natural earth",size="price")

#fig.show()
fig=px.scatter_mapbox(data_frame=df,

                      lat="latitude",

                      lon="longitude",

                      color="price",

                    hover_data=["price"],

                     hover_name="neighbourhood",

                     height=500,

                      width=800,

                     size="price");



fig.update_layout(mapbox_style="open-street-map")

fig.update_layout(margin={"r":0,"t":1,"l":0,"b":0})

fig.show()
df.describe().T
len(df[df.duplicated(['host_id','neighbourhood'],keep=False)].sort_values(['host_id']))
len(df[df.duplicated(['host_id','neighbourhood',"price"],keep=False)].sort_values(['host_id']))
len(df[df.duplicated(['host_id','neighbourhood',"price","host_name"],keep=False)].sort_values(['host_id']))
df.host_id.nunique()
missing_frame=pd.DataFrame(zip(df.isnull().sum(),(df.isnull().sum()/len(df))),columns=["missing_no","missing_rate"])

missing_frame.index=df.columns

missing_frame

df.eq(0).sum().to_frame()
df=df.drop(df[df.price<=0].index)
msno.matrix(df);
msno.bar(df);
msno.heatmap(df);
plt.figure(figsize=(12,5));

sns.heatmap(df.corr(method="spearman").abs(),annot=True);
nominal.associations(df.dropna(),figsize=(20,10),mark_columns=True);
m = folium.Map([40 ,-73], zoom_start=3,width="%100",height="%100")

locations = list(zip(df.latitude, df.longitude))

cluster = plugins.MarkerCluster(locations=locations,popups=df["neighbourhood"].tolist())

m.add_child(cluster)

m
m = folium.Map(location=[40,-73],width="%100",height="%100",zoom_start=3)

for i in range(len(locations)):

    folium.CircleMarker(location=locations[i],radius=1).add_to(m)

m
df.groupby("neighbourhood_group")["price"].mean().sort_values(ascending=True).plot.bar(figsize=(12,5),color="orangered");
sns.catplot(x="room_type",y="price",data=df,hue="neighbourhood_group",aspect=2);
sns.catplot(x="room_type",y="price",data=df[df.price<1000],hue="neighbourhood_group",aspect=3,kind="box");
pd.crosstab(df.neighbourhood_group,df.room_type).plot.bar(figsize=(12,5));
df.groupby("neighbourhood_group")["price"].agg(['mean',"std","median","max","min","count"])
df.groupby("neighbourhood")["price"].agg(['mean',"std","median","max","min","count"]).sort_values(by="mean",ascending=False)[:10]
df.groupby(["room_type","neighbourhood_group"])["price"].agg(['mean',"std","median","max","min","count"]).sort_values(by="max",ascending=False)
sns.catplot(x="host_name",y="price",data=df[df.price>5000],aspect=3,kind="bar");
colors = ['gold', 'lightgreen', 'darkorange']

fig = px.pie(df, names='room_type',title='Room Type Counts')

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,

                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))

fig.show()
sns.FacetGrid(df[df.price<1000],hue="neighbourhood_group",height=5,aspect=2).map(sns.kdeplot,"price",shade=False).add_legend();
sns.catplot(x="neighbourhood_group",y="price",hue="room_type",data=df[df.price<500],aspect=3,kind="violin");
sns.FacetGrid(df[df.price<1000],col="neighbourhood_group",height=3,aspect=2).map(sns.distplot,"price").add_legend();
sns.FacetGrid(df[df.price<1000],col="neighbourhood_group",row="room_type",height=3,aspect=2).map(sns.distplot,"price",color="m").add_legend();
sns.FacetGrid(df[df.price<1000],hue="neighbourhood_group",row="room_type",height=4,aspect=2).map(sns.kdeplot,"price",shade=False).add_legend();
m = folium.Map([40 ,-73], zoom_start=3,width="%100",height="%100",tiles="Stamen Watercolor")

locations = list(zip(df[df.price>5000].latitude, df[df.price>5000].longitude))

cluster = plugins.MarkerCluster(locations=locations,popups=df["neighbourhood"].tolist())

m.add_child(cluster)

m #the most expensive places
fig=px.scatter_mapbox(data_frame=df[df.price>5000],

                      lat="latitude",

                      lon="longitude",

                      color="price",

                    hover_data=["price"],

                     hover_name="neighbourhood",

                     height=500,

                      width=800,

                     size="price");



fig.update_layout(mapbox_style="open-street-map")

fig.update_layout(margin={"r":0,"t":1,"l":0,"b":0})

fig.show()
fig=px.scatter_mapbox(data_frame=df[df.price<100],

                      lat="latitude",

                      lon="longitude",

                      color="price",

                    hover_data=["price"],

                     hover_name="neighbourhood",

                     height=500,

                      width=800,

                     size="price");



fig.update_layout(mapbox_style="open-street-map")

fig.update_layout(margin={"r":0,"t":1,"l":0,"b":0})

fig.show()
df.price.describe()
m = folium.Map(location=[40,-73],width="%100",height="%100")



colormap = cm.StepColormap(colors=['green','yellow','orange','red'] ,index=[10,69,175,327,10000],vmin= 10,vmax=10000)



for loc, p in zip(zip(df["latitude"],df["longitude"]),df["price"]):

    folium.Circle(

        location=loc,

        radius=2,

        fill=True,

        color=colormap(p),

    ).add_to(m)





m