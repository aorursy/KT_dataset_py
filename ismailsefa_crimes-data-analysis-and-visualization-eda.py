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
df=pd.read_csv("../input/crimes-in-boston/crime.csv",encoding = "ISO-8859-1")
df.sample(5)
df.info()
df.isnull().sum()
import missingno as msno

msno.matrix(df)

plt.show()
df.columns
df.drop(columns=['INCIDENT_NUMBER','OFFENSE_CODE','SHOOTING'],inplace=True)
plt.figure(figsize=(25,15))

ax = sns.countplot(x="HOUR", data=df,

                   facecolor=(0, 0, 0, 0),

                   linewidth=5,

                   edgecolor=sns.color_palette("dark", 24))
df2 = pd.DataFrame(columns = ['Offenses'])

df2["Offenses"]=[each for each in df.OFFENSE_CODE_GROUP.unique()]

df2["Count"]=[len(df[df.OFFENSE_CODE_GROUP==each]) for each in df2.Offenses]

df2=df2.sort_values(by=['Count'],ascending=False)



plt.figure(figsize=(25,15))

sns.barplot(x=df2.Offenses.head(50), y=df2.Count.head(50))

plt.xticks(rotation= 90)

plt.xlabel('Offenses')

plt.ylabel('Count')

plt.show()
x = df.DAY_OF_WEEK

y = df.HOUR



fig = go.Figure(go.Histogram2d(

        x=x,

        y=y

    ))

fig.show()
labels = df.DAY_OF_WEEK.unique()

values=[]

for each in labels:

    values.append(len(df[df.DAY_OF_WEEK==each]))



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])

fig.show()
fig = px.scatter_mapbox(df[df["OFFENSE_CODE_GROUP"]=="Service"], lat="Lat", lon="Long", hover_name="HOUR", hover_data=["YEAR", "HOUR"],

                        color_discrete_sequence=["fuchsia"], zoom=10, height=600)

fig.update_layout(mapbox_style="open-street-map")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
vand=df[df["OFFENSE_CODE_GROUP"]=="Service"].iloc[:,11:13]

vand.rename(columns={'Lat':'latitude','Long':'longitude'}, inplace=True)

vand.latitude.fillna(0, inplace = True)

vand.longitude.fillna(0, inplace = True) 



BostonMap=folium.Map(location=[42.5,-71],zoom_start=10)

HeatMap(data=vand, radius=16).add_to(BostonMap)



BostonMap
plt.figure(figsize=(25,15))

wordcloud = WordCloud(

                          background_color='black',

                          width=1920,

                          height=1080

                         ).generate(" ".join(df.OFFENSE_CODE_GROUP))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')

plt.show()