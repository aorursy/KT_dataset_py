import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.offline as ply

import missingno 

import os

print(os.listdir("../input"))
data = pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv')

xyz=pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/noc_regions.csv')


xyz.drop("notes",axis=1,inplace=True)



xyz.head()

xyz.shape
data=pd.merge(data,xyz,on="NOC")



data_orig= data   # making a copy 
data.head()
data.dtypes
data.shape
missingno.matrix(data)
data.isnull().sum()
data_gold=data[data["Medal"]=="Gold"]
data_silver=data[data["Medal"]=="Silver"]
data_bronze=data[data["Medal"]=="Bronze"]
data_none=data[data["Medal"].isnull()]
# lets do some basic analysis of gold section
data_gold.head()
x=data_gold.groupby(by=["Year","region"])["Medal"].count()
x1=pd.DataFrame(x)    # plot 1
# lets print top 5 countries with highest number of golds
data_gold.groupby(by="region")["Medal"].count().sort_values(ascending=False).head().plot(kind="pie")
# lets print top 5 countries with highest number of silver
data_silver.groupby(by="region")["Medal"].count().sort_values(ascending=False).head().plot(kind="pie")
# lets print top 5 countries with highest number of bronze
z=data_bronze.groupby(by="region")["Medal"].count().sort_values(ascending=False).head()

z=pd.DataFrame(z)

z
import plotly.graph_objs as go

from plotly.offline import iplot
dict1 = dict(type = 'choropleth',

           locations = list(z.index),

           locationmode = 'country names',

           colorscale = 'Portland',

           z = list(z.Medal),

           colorbar = {'title': 'No. Of Bronze'})
dict1
layout = dict(geo={'scope':'world'},title="Top Five Countries Having Highest Bronze Medals")

map = go.Figure([dict1], layout)

iplot(map)
data.head()
data.Medal.fillna(0,inplace=True)
data.head()
# No. of medals won by each country
data.head()
medal_count=pd.DataFrame(data.groupby(by="region")["Medal"].count())
medal_count.index
medal_count
dict1 = dict(type = 'choropleth',

           locations = list(medal_count.index),

           locationmode = 'country names',

           colorscale = 'Cividis',

           z = list(medal_count.Medal),

           colorbar = {'title': 'No. Of Medals'})

layout = dict(geo={'scope':'world'},title="Countries with their Medals count")

map = go.Figure([dict1], layout)

iplot(map)
# top 10 countries with highest no. of medals
medal_ten=medal_count.sort_values(by="Medal",ascending=False).head(10)
dict1 = dict(type = 'choropleth',

           locations = list(medal_ten.index),

           locationmode = 'country names',

           colorscale = 'Portland',

           z = list(medal_ten.Medal),

           colorbar = {'title': 'No. Of Medals'})
layout = dict(geo={'scope':'world'},title="Top 10 Countries Having Highest no. of Medals")

map = go.Figure([dict1], layout)

iplot(map)
data.head()
data["Season"]=data["Games"].str.split(" ").str[1]
data.head()
gm=pd.DataFrame(data.groupby(["Season","Sex"])["Medal"].count())

gm.reset_index(inplace=True)
sns.countplot(data["Sex"],hue="Season",data=data)

plt.xlabel("Gender")

plt.ylabel("No. of Medals won")

plt.show()
import plotly.express as px

gm
fig = px.bar(gm,x="Sex",y="Medal",title="No. Of Medals",barmode="group",color="Season")

fig.data[0].marker.line.width = 4

fig.data[0].marker.line.color = "black"

fig.data[1].marker.line.width = 4

fig.data[1].marker.line.color = "black"

fig.show()
data.head()
d1=data[data["Sex"]=="M"]

d1=d1[d1["Medal"] != 0]

d1=pd.DataFrame(d1["Event"].value_counts().head())

d1.reset_index(inplace=True)

d1
fig = px.bar(d1,x="index",y="Event",title="Top 5 Men Events with respect to number of Medals earned")

fig.data[0].marker.line.width = 4

fig.data[0].marker.line.color = "black"

fig.show()
data.head()
# Grouping the events along with their participants with their medals info

data.groupby(by="Sport")["Name"].value_counts().sort_values(ascending=False)
data.head()
# Lets have look at how many males and how many females have taken part¶



Mno=data[data.Sex=="M"]["ID"].count()

Fno=data[data.Sex=="F"]["ID"].count()



labels = ["Male Participants","Female Participants"]

values = [Mno,Fno]

title_text="Ratio Of Male and Female Participants",



fig = go.Figure(data=[go.Pie(labels=labels, values=values,hole=0.5)])

fig.show()
# Lets see which are the sports USA have won maximum Gold Medals (Showing Top 5 Sports)

z=pd.DataFrame(data[(data["region"]=="USA") & (data["Medal"]=="Gold")]["Sport"].value_counts().head())

z = z.reset_index()
fig = go.Figure(go.Funnel(

    y =list(z["index"]),

    x = list(z["Sport"])))



fig.show()
f=pd.DataFrame(data[(data["region"]=="USA") & (data["Medal"]=="Gold")&(data["Sex"]=="F")]["Sport"].value_counts().head())

f = f.reset_index()

f["Sport_F"]=f["Sport"]

f.drop("Sport",axis=1,inplace=True)

f
m=pd.DataFrame(data[(data["region"]=="USA") & (data["Medal"]=="Gold")&(data["Sex"]=="M")]["Sport"].value_counts().head())

m = m.reset_index()

m["Sport_M"]=m["Sport"]

m.drop("Sport",axis=1,inplace=True)

m
m["Sport_F"]=f["Sport_F"]

usa_stats=m

usa_stats
usa_stats["Total"]=usa_stats["Sport_M"]+usa_stats["Sport_F"]

usa_stats
# USA Top 5 Sports having gold medals (Showing Top 5 Sports)



fig = go.Figure()



fig.add_trace(go.Funnel(

    name = 'Gold_Male_USA',

    y =list(m["index"]),

    x = [usa_stats["Sport_M"][0],usa_stats["Sport_M"][1],usa_stats["Sport_M"][2],usa_stats["Sport_M"][3],usa_stats["Sport_M"][4]],

    textinfo = "value"))



fig.add_trace(go.Funnel(

    name = 'Gold_Female_USA',

    orientation = "h",

    y =list(m["index"]),

    x = [usa_stats["Sport_F"][0],usa_stats["Sport_F"][1],usa_stats["Sport_F"][2],usa_stats["Sport_F"][3],usa_stats["Sport_F"][4]],

    textposition = "inside",



    textinfo = "value"))





fig.show()






