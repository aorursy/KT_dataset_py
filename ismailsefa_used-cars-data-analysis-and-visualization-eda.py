import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# visualization tools

import matplotlib.pyplot as plt

from wordcloud import WordCloud

import seaborn as sns

import plotly.graph_objects as go

import plotly.express as px

import folium

from folium.plugins import HeatMap

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv("../input/dataset/craigslistVehicles.csv")
df.sample(5)
df.info()
df.isnull().sum()
import missingno as msno

msno.matrix(df)

plt.show()
df.columns
df.drop(columns=['url','image_url','VIN'],inplace=True)
df=df.sort_values(by=['odometer'],ascending=False)

plt.figure(figsize=(25,15))

sns.barplot(x=df.manufacturer, y=df.odometer)

plt.xticks(rotation= 90)

plt.xlabel('Manufacturer')

plt.ylabel('Odometer')

plt.show()
gasLabels = df[df["fuel"]=="gas"].paint_color.value_counts().head(10).index

gasValues = df[df["fuel"]=="gas"].paint_color.value_counts().head(10).values

dieselLabels = df[df["fuel"]=="diesel"].paint_color.value_counts().head(10).index

dieselValues = df[df["fuel"]=="diesel"].paint_color.value_counts().head(10).values

electricLabels = df[df["fuel"]=="electric"].paint_color.value_counts().head(10).index

electricValues = df[df["fuel"]=="electric"].paint_color.value_counts().head(10).values



from plotly.subplots import make_subplots



# Create subplots: use 'domain' type for Pie subplot

fig = make_subplots(rows=1, cols=3, specs=[[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}]])

fig.add_trace(go.Pie(labels=gasLabels, values=gasValues, name="Gas Car"),

              1, 1)

fig.add_trace(go.Pie(labels=dieselLabels, values=dieselValues, name="Diesel Car"),

              1, 2)

fig.add_trace(go.Pie(labels=electricLabels, values=electricValues, name="Electric Car"),

              1, 3)



# Use `hole` to create a donut-like pie chart

fig.update_traces(hole=.4, hoverinfo="label+percent+name")



fig.show()
x = df.type

y = df.paint_color



fig = go.Figure(go.Histogram2d(

        x=x,

        y=y

    ))

fig.show()
fig = px.scatter_mapbox(df[df["type"]=="bus"], lat="lat", lon="long", hover_name="paint_color", hover_data=["paint_color", "price"],

                        color_discrete_sequence=["fuchsia"], zoom=4, height=600)

fig.update_layout(mapbox_style="open-street-map")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
cars=df[df["type"]=="bus"].iloc[:,17:19]

cars.rename(columns={'lat':'latitude','long':'longitude'}, inplace=True)

cars.latitude.fillna(0, inplace = True)

cars.longitude.fillna(0, inplace = True) 



CarMap=folium.Map(location=[42.5,-71],zoom_start=4)

HeatMap(data=cars, radius=16).add_to(CarMap)

CarMap.save('index.html')

CarMap