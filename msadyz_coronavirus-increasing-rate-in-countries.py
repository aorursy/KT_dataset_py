# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Visualisation libraries

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()

from plotly.offline import init_notebook_mode, iplot 

import plotly.express as px

import plotly.offline as py

from plotly.subplots import make_subplots

import plotly.graph_objects as go



import pycountry

py.init_notebook_mode(connected=True)

import folium 

from folium import plugins





# Graphics in retina format 

%config InlineBackend.figure_format = 'retina' 



# Increase the default plot size and set the color scheme

plt.rcParams['figure.figsize'] = 8, 5

#plt.rcParams['image.cmap'] = 'viridis'





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Disable warnings 

import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv', parse_dates=["Last Update", "Date"])

data = data.set_index("Date").drop(columns=["Sno"])

data
df = data.resample("D").sum()

df
fig = go.Figure(data=[

    go.Bar(name='Confirmed', x=df.index, y=df.Confirmed),

    go.Bar(name='Deaths', x=df.index, y=df.Deaths),

    go.Bar(name='Recovered', x=df.index, y=df.Recovered)

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.show()
fig = make_subplots(rows=1, cols=3, specs=[[{"type" : "pie"}, {"type" : "pie"},{"type" : "pie"}]],

                    subplot_titles=("number of provience in countries", "Deaths", "Recovers"))



fig.add_trace(

    go.Pie(labels=data.groupby('Country')['Province/State'].nunique().sort_values(ascending=False)[:10].index,

           values=data.groupby('Country')['Province/State'].nunique().sort_values(ascending=False)[:10].values,

           rotation=45),

    row=1, col=1

)



fig.add_trace(

    go.Pie(labels=data[data.Deaths > 0].groupby('Country')["Deaths"].max().index,

           values=data[data.Deaths > 0].groupby('Country')["Deaths"].max().values, 

           rotation=45),

    row=1, col=2, 

)

fig.add_trace(

    go.Pie(labels=data.groupby('Country')["Recovered"].max().sort_values(ascending=False).index[:5],

           values=data.groupby('Country')["Recovered"].max().sort_values(ascending=False).values[:5],

           rotation=45),

    row=1, col=3,

)



fig.update_layout(height=400, showlegend=True)

fig.show()
import plotly.graph_objects as go



days = df.index



fig = go.Figure()

fig.add_trace(go.Bar(

    x=days,

    y=df['Confirmed'],

    name='Confirmed per day',

    marker_color='yellow'

))





fig.add_trace(go.Bar(

    x=days,

    y=df['Deaths'],

    name='Deaths per day',

    marker_color='red'

))



fig.add_trace(go.Bar(

    x=days,

    y=df["Recovered"],

    name='Recovered per day',

    marker_color='green'

))



# Here we modify the tickangle of the xaxis, resulting in rotated labels.

fig.update_layout(barmode='group', xaxis_tickangle=-45)

fig.show()
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36",

                      timeout=20)
d={'name': ['Australia',

  'Belgium',

  'Brazil',

  'Cambodia',

  'Canada',

  'China',

  'Finland',

  'France',

  'Germany',

  'Hong Kong',

  'India',

  'Italy',

  'Ivory Coast',

  'Japan',

  'Macau',

  'Mainland China',

  'Malaysia',

  'Mexico',

  'Nepal',

  'Philippines',

  'Russia',

  'Singapore',

  'South Korea',

  'Spain',

  'Sri Lanka',

  'Sweden',

  'Taiwan',

  'Thailand',

  'UK',

  'US',

  'United Arab Emirates',

  'Vietnam'],

 'lat': np.array([-24.7761086,  50.6402809, -10.3333333,  13.5066394,  61.0666922,

         35.000074 ,  63.2467777,  46.603354 ,  51.0834196,  22.2793278,

         22.3511148,  42.6384261,   7.9897371,  36.5748441,  22.1757605,

         30.5951051,   4.5693754,  19.4326009,  28.1083929,  12.7503486,

         64.6863136,   1.357107 ,  36.5581914,  39.3262345,   7.5554942,

         59.6749712,  23.9739374,  14.8971921,  54.7023545,  39.7837304,

         24.0002488,  13.2904027]),

 'lon': np.array([ 134.755    ,    4.6667145,  -53.2      ,  104.869423 ,

        -107.9917071,  104.999927 ,   25.9209164,    1.8883335,

          10.4234469,  114.1628131,   78.6677428,   12.674297 ,

          -5.5679458,  139.2394179,  113.5514142,  114.2999353,

         102.2656823,  -99.1333416,   84.0917139,  122.7312101,

          97.7453061,  103.8194992,  127.9408564,   -4.8380649,

          80.7137847,   14.5208584,  120.9820179,  100.83273  ,

          -3.2765753, -100.4458825,   53.9994829,  108.4265113])}



index = d.pop("name")

wd = pd.DataFrame(d, index=index)
dfc = data.groupby("Country").max()

dfc
world_data = pd.concat((wd, dfc), axis=1)

world_data["ConfiremedRatio"] = world_data.Confirmed/(world_data.Confirmed.sum())



world_data
con_vals = world_data.Confirmed.values.copy()

con_vals.sort()
# create map and display itconf

world_map = folium.Map(location=[10, -20], zoom_start=2.3,)   # tiles='Stamen Toner' or 'Stamen Terrain'



for lat, lon, value,ratio, name in zip(world_data['lat'], world_data['lon'], 

                                 world_data['Confirmed'], world_data.ConfiremedRatio,

                                 world_data.index):

    rad = value

    fo = 0.4

    if value==0:

        continue

    if value in con_vals[-2:]:

        rad/=300 if value == con_vals[-1] else 30

        fo = ratio*2/3 if value == con_vals[-1] else fo

    folium.CircleMarker([lat, lon],

                        radius=rad,

                        popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'

                                '<strong>Confirmed Cases from 22nd of jan</strong>: ' + str(value) + '<br>'),

                        color='orange',

                        

                        fill_color='red',

                        fill_opacity=fo).add_to(world_map)

world_map

#world_map.save('countries_affected.html')