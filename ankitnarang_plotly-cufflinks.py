!pip install plotly

!pip install cufflinks

!pip install foliu
!pip install mpl_toolkits
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
from plotly.offline import iplot

import plotly as py

import plotly.tools as tls
import cufflinks as cf
print(py.__version__)
# import the library

import folium

import pandas as pd

 

# Make a data frame with dots to show on the map

data = pd.DataFrame({

   'lat':[-58, 2, 145, 30.32, -4.03, -73.57, 36.82, -38.5],

   'lon':[-34, 49, -38, 59.93, 5.33, 45.52, -1.29, -12.97],

   'name':['Buenos Aires', 'Paris', 'melbourne', 'St Petersbourg', 'Abidjan', 'Montreal', 'Nairobi', 'Salvador'],

   'value':[10,12,40,70,23,43,100,43]

})

data

 

# Make an empty map

m = folium.Map(location=[20,0], tiles="Mapbox Bright", zoom_start=2)

 

# I can add marker one by one on the map

for i in range(0,len(data)):

   folium.Circle(

      location=[data.iloc[i]['lon'], data.iloc[i]['lat']],

      popup=data.iloc[i]['name'],

      radius=data.iloc[i]['value']*10000,

      color='crimson',

      fill=True,

      fill_color='crimson'

   ).add_to(m)

 

# Save it as html

#m.save('mymap.html')

from urllib.request import urlopen

import json

with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:

    counties = json.load(response)



import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/fips-unemp-16.csv",

                   dtype={"fips": str})



import plotly.graph_objects as go



fig = go.Figure(go.Choroplethmapbox(geojson=counties, locations=df.fips, z=df.unemp,

                                    colorscale="Viridis", zmin=0, zmax=12,

                                    marker_opacity=0.5, marker_line_width=0))

fig.update_layout(mapbox_style="carto-positron",

                  mapbox_zoom=3, mapbox_center = {"lat": 37.0902, "lon": -95.7129})

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
py.offline.init_notebook_mode(connected=True)
cf.go_offline()
df = pd.DataFrame(np.random.randn(100,3), columns = ['A', 'B', 'C'])

df.head()

df['A'] = df['A'].cumsum() + 20

df['B'] = df['B'].cumsum() + 20

df['C'] = df['C'].cumsum() + 20
import pandas as pd

us_cities = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/us-cities-top-1k.csv")



import plotly.express as px



fig = px.scatter_mapbox(us_cities, lat="lat", lon="lon", hover_name="City", hover_data=["State", "Population"],

                        color_discrete_sequence=["fuchsia"], zoom=3, height=300)

fig.update_layout(mapbox_style="open-street-map")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
df.head()
df.iplot()
plt.plot(df)
df.plot()
df.iplot(x = 'A', y = 'B', mode = 'markers', size = 25, color="green")
titanic = sns.load_dataset('titanic')

titanic.head()
titanic.iplot(kind = "bar", x = "sex", y="survived", title="Survived", xTitle = "Sex", yTitle='#Survived')
titanic['sex'].value_counts()
cf.getThemes()

cf.set_config_file(theme='polar')

df.iplot(kind = 'bar', barmode='stack', bargap=0.5)
df.iplot(kind = 'bar', barmode='stack', bargap=0.5)
df.iplot(kind = 'barh', barmode='stack', bargap=0.5)
1,2,3,4,5,6,7
df.iplot(kind = 'box')
df.iplot()
df.iplot(kind = 'area')
df3 = pd.DataFrame({'X': [10,20,30,20,10], 'Y': [10, 20, 30, 20, 10], 'Z': [10, 20, 30, 20, 10]})

df3.head()
df3.iplot(kind='surface', colorscale='rdylbu')
help(cf.datagen)
cf.datagen.sinwave(10, 0.25).iplot(kind = 'surface')
cf.datagen.scatter3d(2, 150, mode = 'stocks').iplot(kind = 'scatter3d', x = 'x', y= 'y', z = 'z')
import numpy

df[['A', 'B']].iplot(kind = 'spread')
df.iplot(kind='hist', bins = 25, barmode = 'overlay', bargap=0.5)
cf.datagen.bubble3d(5,4,mode='stocks').iplot(kind='bubble3d',x='x',y='y',z='z', size='size')
cf.datagen.heatmap(20,20).iplot(kind = 'heatmap', colorscale='spectral', title='Cufflinks - Heatmap')