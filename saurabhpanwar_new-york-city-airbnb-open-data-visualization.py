# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.offline as pyo

import plotly.graph_objs as go

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px



%matplotlib inline

from ipywidgets import widgets





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
df.head()
df.info()
df= df.drop(df[(df['minimum_nights']>365)].index)

df= df.drop(df[(df['number_of_reviews']>500)].index)

df= df.drop(df[(df['calculated_host_listings_count']>100)].index)

df= df.drop(df[(df['price']>1800)].index)

df= df.drop(df[(df['price']<1)].index)



df.describe()


df.neighbourhood_group.unique()
neighbour_group_df =df.pivot_table('price', ['neighbourhood_group'], aggfunc='mean').reset_index()
fig = px.bar(neighbour_group_df, x='neighbourhood_group', y='price',

             hover_data=['price'], color='price',  barmode ='relative',

             labels={'pop':'Neighbourhood group and their pricing'}, height=400, width=800)



fig.show()


fig = px.histogram(df, x="neighbourhood_group", color = 'neighbourhood_group', height=600, width=800, )

fig.update_layout(showlegend = True)

fig.show()
roomdf = df.groupby('room_type').size()/df['room_type'].count()*100

labels = roomdf.index

values = roomdf.values



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])

fig.show()
neighbour_df =df.pivot_table(['price', 'number_of_reviews', 'calculated_host_listings_count', 'neighbourhood_group' ] , ['neighbourhood'], aggfunc='mean').reset_index()
fig = px.scatter(neighbour_df, x="neighbourhood", y="price", color="calculated_host_listings_count",

                 size='price', height=500, width=800)

fig.update_layout(showlegend = False)

fig.show()
fig = px.histogram(df, x="price", color = 'neighbourhood_group',marginal="rug",  hover_data=df.columns, height=600, width=800, )

fig.update_layout(showlegend = False)

fig.show()
airbnb_100 = df.nsmallest(200,'price')

fig = px.scatter(airbnb_100, x="host_name", y="reviews_per_month", color="price", size = 'calculated_host_listings_count', height=500, width=800)

fig.update_layout(showlegend = False)

fig.show()
large_airbnb_200 = df.nlargest(200,'price')

fig = px.scatter(large_airbnb_200, x="price", y="reviews_per_month", color="neighbourhood_group", size = 'calculated_host_listings_count', 

                 hover_data=large_airbnb_200.columns, height=500, width=800)

fig.update_layout(showlegend = False)

fig.show()
import plotly.express as px



mapbox_access_token = 'pk.eyJ1IjoiYmlkZHkiLCJhIjoiY2pxNWZ1bjZ6MjRjczRhbXNxeG5udzkyNSJ9.xX6QLOAcoBmXZdUdocAeuA'

px.set_mapbox_access_token(mapbox_access_token)

fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", color="neighbourhood_group", size = 'price', opacity= 0.8,

                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=16, zoom=9.2,height=400, width=800 )

fig.update_layout(

    mapbox_style="white-bg",

    showlegend = False,

    mapbox_layers=[

        {

            "below": 'traces',

            "sourcetype": "raster",

            "source": [

                "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"

            ]

        },

      ]

)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":4})

fig.show()