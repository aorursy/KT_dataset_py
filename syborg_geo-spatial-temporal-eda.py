import random

import plotly

import numpy as np

import pandas as pd

from pathlib import Path

import plotly.graph_objs as go

import matplotlib.pyplot as plt
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

mapbox_access_token = user_secrets.get_secret("mapbox_access_token")
path = Path('/kaggle/input/birdsong-recognition')

train = pd.read_csv(path/'train.csv')
train.info()
train['_time'] = pd.to_datetime(train.time, errors='coerce').dt.strftime('%H:%M:%S')

train['_date'] = pd.to_datetime(train.date, format='%Y-%m-%d %H:%M:%S', errors='coerce').dt.strftime('%Y-%m-%d')

# creating a new column: _datetime

train['_datetime'] = pd.to_datetime(train['_date'] + ' ' + train['_time'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
train[train._datetime.isna()][['date', 'time', '_datetime']].head(10)
fig = go.Figure(data=[go.Histogram(x=pd.to_datetime(train._time, format='%H:%M:%S').dt.round('15min'))]) # rounding to nearest quarter of an hour

fig.show()
print(train._time.isna().sum())

train[train._time.isna()]['time'].unique()
print(train._date.isna().sum())

train[train._date.isna()]['date'].unique()
train['_year_month'] = train.date.apply(lambda x : '-'.join(x.split('-')[:2])) # 'keeping only year-month and excluding date'

train['_year_month'] = pd.to_datetime(train._year_month, format='%Y-%m', errors='coerce')

train._year_month.isna().sum()
fig = go.Figure(data=[go.Histogram(x=train._year_month)])

fig.show()
fig = go.Figure(data=[go.Histogram(x=pd.DatetimeIndex(train._year_month).month)])

fig.show()
fig = go.Figure(data=[go.Histogram(x=pd.DatetimeIndex(train._year_month).year)])

fig.show()
train['_year_month'] = train._year_month.dt.strftime('%Y-%m') # converting to string 

train['_elevation'] = train.elevation.apply(lambda x : x.replace('m', '').replace('~', '').replace(',', '').replace('?', '').strip()) # replace

train.loc[train._elevation.isin(['1650-1900', '930-990', 'Unknown', '-']), '_elevation'] = '' # assign empty string 

df = train.loc[(train.longitude != 'Not specified') & (train.latitude != 'Not specified'), ['country', 'latitude', 'longitude', '_elevation', '_year_month', 'ebird_code', 'elevation']]

df.loc[df._elevation == '', '_elevation'] = None # empty string with None

df['_elevation'] = df._elevation.astype(float) # convert to float

df['_elevation'].fillna(0.0, inplace=True) # replace NaN with 0.0

df['_elevation'] = (df._elevation + 100.0)/80.0 # scale values 
df.info()
df = df.loc[~df._year_month.isna(), :] # dropping all NaN dates
df = df.set_index('_year_month') # setting date as the dataframe index
df.head()
# total no of birds

number_of_colors = 264



# list of random hex-valued colors 

color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])

             for i in range(number_of_colors)]



ebird_code = df.ebird_code.unique().tolist()

# get ID and color for each bird

EBIRD_CODE = {k : color[i] for i, k in enumerate(ebird_code)}

# assign them to the dataframe

df['_color'] = df.ebird_code.apply(lambda x : EBIRD_CODE[x])
months = sorted(df.index.unique().tolist())
frames = [{   

    'name':'frame_{}'.format(x),

    'data':[{

        'type':'scattermapbox',

        'lat':np.array(df.xs(x)['latitude']),

        'lon':np.array(df.xs(x)['longitude']),

        'marker':go.scattermapbox.Marker(

            size= 9 + df.xs(x)['_elevation'],

            color=df.xs(x)['_color']

        ),

        'customdata': np.stack((df.xs(x)['ebird_code'], df.xs(x)['elevation'], df.xs(x)['country']), axis=-1),

        'hovertemplate': "<extra></extra> 🐦 <em>%{customdata[0]}</em><br> 📏 %{customdata[1]}<br> 🗺️ %{customdata[2]}<br>",

    }],           

} for x in months]
sliders = [{

    'transition':{'duration': 0},

    'x':0.08, 

    'len':0.88,

    'currentvalue':{'font':{'size':15}, 'prefix':'📅 ', 'visible':True, 'xanchor':'center'},  

    'steps':[

        {

            'label':x,

            'method':'animate',

            'args':[

                ['frame_{}'.format(x)],

                {'mode':'immediate', 'frame':{'duration':100, 'redraw': True}, 'transition':{'duration':50}}

              ],

        } for x in months]

}]
play_button = [

    {

        "buttons": [

            {

                "args": [None, {"frame": {"duration": 100, "redraw": True},

                                "fromcurrent": True, "transition": {"duration": 50}}],

                "label": "Play",

                "method": "animate"

            },

            {

                "args": [[None], {"frame": {"duration": 0, "redraw": False},

                                  "mode": "immediate",

                                  "transition": {"duration": 0}}],

                "label": "Pause",

                "method": "animate"

            }

        ],

        "direction": "left",

        "pad": {"r": 10, "t": 87},

        "showactive": True,

        "type": "buttons",

        "x": 0.1,

        "xanchor": "right",

        "y": 0,

        "yanchor": "top"

    }

]
# defining the initial state

data = frames[0]['data']



# adding all sliders and play button to the layout

layout = go.Layout(

    sliders=sliders,

    updatemenus=play_button,

    title="Birds over the years",

    mapbox={

        'accesstoken':mapbox_access_token,

        'center':{"lat": 37.86, "lon": 2.15},

        'zoom':1.7,

        'style':'dark', # choose from: dark or light

    },

    height=1000

)



# creating the figure

fig = go.Figure(data=data, layout=layout, frames=frames)



# displaying the figure

fig.show()