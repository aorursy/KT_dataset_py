# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# import warnings
import warnings
# ignore warnings
warnings.filterwarnings("ignore")
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
world = pd.read_csv("../input/countries of the world.csv")
world.head(10)
world.info()
world.dtypes
world.columns = (["country","region","population","area","density","coastline","migration","infant_mortality","gdp","literacy","phones","arable","crops","other","climate","birthrate","deathrate","agriculture","industry","service"])
world.country = world.country.astype('category')
world.region = world.region.astype('category')
world.density = world.density.str.replace(",",".").astype(float)
world.coastline = world.coastline.str.replace(",",".").astype(float)
world.migration = world.migration.str.replace(",",".").astype(float)
world.infant_mortality = world.infant_mortality.str.replace(",",".").astype(float)
world.literacy = world.literacy.str.replace(",",".").astype(float)
world.phones = world.phones.str.replace(",",".").astype(float)
world.arable = world.arable.str.replace(",",".").astype(float)
world.crops = world.crops.str.replace(",",".").astype(float)
world.other = world.other.str.replace(",",".").astype(float)
world.climate = world.climate.str.replace(",",".").astype(float)
world.birthrate = world.birthrate.str.replace(",",".").astype(float)
world.deathrate = world.deathrate.str.replace(",",".").astype(float)
world.agriculture = world.agriculture.str.replace(",",".").astype(float)
world.industry = world.industry.str.replace(",",".").astype(float)
world.service = world.service.str.replace(",",".").astype(float)

world.info()
missing = world.isnull().sum()
missing
world.fillna(world.mean(),inplace=True)
world.region = world.region.str.strip()
group = world.groupby("region")
group.mean()
world.head(10)
region = world.region.value_counts()
plt.figure(figsize=(10,7))
sns.barplot(x=region.index,y=region.values)
plt.xticks(rotation=45)
plt.ylabel('Number of countries')
plt.xlabel('Region')
plt.title('Number of Countries by REGİON',color = 'red',fontsize=20)
sns.set(style="darkgrid",font_scale=1.5)
f, axes = plt.subplots(2,2,figsize=(15,10))

sns.distplot(world.infant_mortality,bins=20,kde=False,color="y",ax=axes[0,0])
sns.distplot(world.gdp,hist=False,rug=True,color="r",ax=axes[0,1])
sns.distplot(world.birthrate,hist=False,color="g",kde_kws={"shade":True},ax=axes[1,0])
sns.distplot(world.deathrate,color="m",ax=axes[1,1])
sns.boxplot(x="region",y="gdp",data=world,width=0.7,palette="Set3",fliersize=5)
plt.xticks(rotation=90)
plt.title("GDP BY REGİON",color="red")
world.corr()
f,ax = plt.subplots(figsize=(18, 16))
sns.heatmap(world.corr(), annot=True, linewidths=.8, fmt= '.1f',ax=ax)
x = world.loc[:,["region","gdp","infant_mortality","birthrate","phones","literacy","service"]]
sns.pairplot(x,hue="region",palette="inferno")
sns.lmplot(x="gdp",y="phones",data=world,height=10)
sns.lmplot(x="gdp",y="service",data=world,height=10)
gdp=world.sort_values(["gdp"],ascending=False)
# prepare data frame
df = gdp.iloc[:100,:]

# Creating trace1
trace1 = go.Scatter(
                    x = df.gdp,
                    y = df.birthrate,
                    mode = "lines",
                    name = "Birthrate",
                    marker = dict(color = 'rgba(235,66,30, 0.8)'),
                    text= df.country)
# Creating trace2
trace2 = go.Scatter(
                    x = df.gdp,
                    y = df.deathrate,
                    mode = "lines+markers",
                    name = "Deathrate",
                    marker = dict(color = 'rgba(10,10,180, 0.8)'),
                    text= df.country)
z = [trace1, trace2]
layout = dict(title = 'Birthrate and Deathrate of World Countries (Top 100)',
              xaxis= dict(title= 'GDP',ticklen= 5,zeroline= False)
             )
fig = dict(data = z, layout = layout)
iplot(fig)
# prepare data frame
df = gdp.iloc[77:177,:]

# Creating trace1
trace1 = go.Scatter(
                    x = df.gdp,
                    y = df.birthrate,
                    mode = "lines",
                    name = "Birthrate",
                    marker = dict(color = 'rgba(235,66,30, 0.8)'),
                    text= df.country)
# Creating trace2
trace2 = go.Scatter(
                    x = df.gdp,
                    y = df.deathrate,
                    mode = "lines+markers",
                    name = "Deathrate",
                    marker = dict(color = 'rgba(10,10,180, 0.8)'),
                    text= df.country)
z = [trace1, trace2]
layout = dict(title = 'Birthrate and Deathrate Percentage of World Countries (Last 100)',
              xaxis= dict(title= 'GDP',ticklen= 5,zeroline= False)
             )
fig = dict(data = z, layout = layout)
iplot(fig)
# prepare data frame
df = gdp.iloc[:100,:]

# Creating trace1
trace1 = go.Scatter(
                    x = df.gdp,
                    y = df.agriculture,
                    mode = "lines+markers",
                    name = "AGRICULTURE",
                    marker = dict(color = 'rgba(235,66,30, 0.8)'),
                    text= df.country)
# Creating trace2
trace2 = go.Scatter(
                    x = df.gdp,
                    y = df.industry,
                    mode = "lines+markers",
                    name = "INDUSTRY",
                    marker = dict(color = 'rgba(10,10,180, 0.8)'),
                    text= df.country)
# Creating trace3
trace3 = go.Scatter(
                    x = df.gdp,
                    y = df.service,
                    mode = "lines+markers",
                    name = "SERVICE",
                    marker = dict(color = 'rgba(10,250,60, 0.8)'),
                    text= df.country)


z = [trace1, trace2,trace3]
layout = dict(title = 'Service , Industry and Agriculture Percentage of World Countries (TOP 100)',
              xaxis= dict(title= 'GDP',ticklen= 5,zeroline= False)
             )
fig = dict(data = z, layout = layout)
iplot(fig)
# prepare data frame
df = gdp.iloc[77:,:]

# Creating trace1
trace1 = go.Scatter(
                    x = df.gdp,
                    y = df.agriculture,
                    mode = "lines+markers",
                    name = "AGRICULTURE",
                    marker = dict(color = 'rgba(235,66,30, 0.8)'),
                    text= df.country)
# Creating trace2
trace2 = go.Scatter(
                    x = df.gdp,
                    y = df.industry,
                    mode = "lines+markers",
                    name = "INDUSTRY",
                    marker = dict(color = 'rgba(10,10,180, 0.8)'),
                    text= df.country)
# Creating trace3
trace3 = go.Scatter(
                    x = df.gdp,
                    y = df.service,
                    mode = "lines+markers",
                    name = "SERVICE",
                    marker = dict(color = 'rgba(10,250,60, 0.8)'),
                    text= df.country)


z = [trace1, trace2,trace3]
layout = dict(title = 'Service , Industry and Agriculture Percentage of World Countries (LAST 100)',
              xaxis= dict(title= 'GDP',ticklen= 5,zeroline= False)
             )
fig = dict(data = z, layout = layout)
iplot(fig)
lit = world.sort_values("literacy",ascending=False).head(7)
trace1 = go.Bar(
                x = lit.country,
                y = lit.agriculture,
                name = "agriculture",
                marker = dict(color = 'rgba(255, 20, 20, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = lit.gdp)
trace2 = go.Bar(
                x = lit.country,
                y = lit.service,
                name = "service",
                marker = dict(color = 'rgba(20, 20, 128, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = lit.gdp)
data = [trace1, trace2]
layout = go.Layout(barmode = "group")
fig = go.Figure(data = data, layout = layout)
iplot(fig)
x = lit.country

trace1 = {
  'x': x,
  'y': lit.industry,
  'name': 'industry',
  'type': 'bar'
};
trace2 = {
  'x': x,
  'y': lit.service,
  'name': 'service',
  'type': 'bar'
};
data = [trace1, trace2];
layout = {
  'xaxis': {'title': 'Top 7 country'},
  'barmode': 'relative',
  'title': 'industry and service percentage of top 7 country (literacy)'
};
fig = go.Figure(data = data, layout = layout)
iplot(fig)
fig = {
  "data": [
    {
      "values": lit.gdp,
      "labels": lit.country,
      "domain": {"x": [0, .5]},
      "name": "GDP percentage of",
      "hoverinfo":"label+percent+name",
      "hole": .3,
      "type": "pie"
    },],
  "layout": {
        "title":"GDP of top 7 country(literacy)",
        "annotations": [
            { "font": { "size": 20},
              "showarrow": False,
              "text": "GDP",
                "x": 0.22,
                "y": 0.5
            },
        ]
    }
}
iplot(fig)
lite = world.sort_values("literacy",ascending=False).head(15)
data = [
    {
        'y': lite.service,
        'x': lite.index,
        'mode': 'markers',
        'marker': {
            'color': lite.service,
            'size': lite.literacy,
            'showscale': True
        },
        "text" :  lite.country    
    }
]
iplot(data)
#Population per country
data = dict(type='choropleth',
locations = world.country,
locationmode = 'country names', z = world.population,
text = world.country, colorbar = {'title':'Population'},
colorscale = 'Blackbody', reversescale = True)
layout = dict(title='Population per country',
geo = dict(showframe=False,projection={'type':'natural earth'}))
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)
#Population per country
data = dict(type='choropleth',
locations = world.country,
locationmode = 'country names', z = world.infant_mortality,
text = world.country, colorbar = {'title':'Infant Mortality'},
colorscale = 'YlOrRd', reversescale = True)
layout = dict(title='Infant Mortality per Country',
geo = dict(showframe=False,projection={'type':'natural earth'}))
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)
#Population per country
data = dict(type='choropleth',
locations = world.country,
locationmode = 'country names', z = world.gdp,
text = world.country, colorbar = {'title':'GDP'},
colorscale = 'Hot', reversescale = True)
layout = dict(title='GDP of World Countries',
geo = dict(showframe=False,projection={'type':'natural earth'}))
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)
