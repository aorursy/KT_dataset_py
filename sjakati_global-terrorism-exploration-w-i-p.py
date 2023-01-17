import os

os.getcwd()



import sys

sys.path.append('/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages')
#imports

import numpy as np

import pandas as pd 

from scipy import stats

import matplotlib.pyplot as plt

import plotly

from plotly.offline import iplot

plotly.offline.init_notebook_mode()

import plotly.graph_objs as go

from plotly.graph_objs import *

from collections import defaultdict

import seaborn as sns
main_df = pd.read_csv('../input/globalterrorismdb_0617dist.csv',encoding='ISO-8859-1')

main_df.head()
print(main_df.info())
##keep track of all the types of attacks and their total counts

def parse_attack_types(arr):

    results = stats.itemfreq(arr['attacktype1_txt'])

    return results

##display the counts of attacks using plotly https://www.plot.ly

attack_type_counts = parse_attack_types(main_df)

_x = []

_y = []

for i in range(0, len(attack_type_counts)):

    _x.append(attack_type_counts[i][0])

    _y.append(attack_type_counts[i][1])



data = [go.Bar(

    x = _x, #names of attacks

    y = _y, #counts of attacks

)]

iplot(data, filename='basic-bar')
##keep track of all the countries and attack counts

def parse_country_attacks(arr):

    results = stats.itemfreq(arr['country_txt'])

    return results

##display the counts of attacks using plotly https://www.plot.ly

country_attack_counts = parse_country_attacks(main_df)

_x = []

_y = []

for i in range(0, len(country_attack_counts)):

    _x.append(country_attack_counts[i][0])

    _y.append(country_attack_counts[i][1])



data = [go.Bar(

    x = _x, #names of attacks

    y = _y, #counts of attacks

)]

iplot(data, filename='basic-bar')
mapbox_access_token = 'pk.eyJ1Ijoic2hpc2hpcmpha2F0aSIsImEiOiJjajhueW8xaHgxZHoyMndydG94aWRkaGlhIn0.hpXRZtotwg3nlsABMBPuYA'

locations = np.dstack((main_df['latitude'], main_df['longitude'], main_df['country_txt']))[0]

data = []

data.append(Scattermapbox(

    lat = main_df['latitude'],

    lon = main_df['longitude'],

    mode = 'markers',

    marker = Marker(

        size=2,

        color = 'rgb(244, 66, 66)'

    ),

    text = main_df['country_txt']

))

layout = Layout(

    autosize=True,

    hovermode='closest',

    mapbox=dict(

        accesstoken=mapbox_access_token,

        bearing=0,

        center=dict(

            lat=45,

            lon=45

        ),

        pitch=0,

        zoom=1,

    ),

)



fig = dict(data=data, layout=layout)

iplot(fig, filename='Montreal Mapbox')

attack_year_type = []

attack_types = []

year_type_counter = defaultdict(lambda: defaultdict(int))

for _, event in main_df.iterrows():

    attack_year_type.append((event['iyear'], event['attacktype1_txt']))

    year_type_counter[int(event['iyear'])][event['attacktype1_txt']] += 1

    if event['attacktype1_txt'] not in attack_types:

        attack_types.append(event['attacktype1_txt']) 
data = []

colors = ['rgb(205, 12, 24)', 'rgb(241, 244, 66)', 'rgb(65, 211, 244)', 'rgb(159, 44, 165)', 'rgb(63, 124, 37)', 'rgb(183, 91, 0)', 'rgb(178, 74, 104)']

def plot_attack(attack_type):

    _x = np.arange(1970,2016)

    _y = []

    

    i = 0

    for year in year_type_counter:

        _y.append(year_type_counter[year][attack_type])

        

    trace = go.Scatter(

        x = _x,

        y = _y,

        name = attack_type,

        line = dict(

        width = 4,

        dash = 'dot')

    )

    data.append(trace)

    i += 1





for attack_type in attack_types:

    plot_attack(attack_type)

layout = dict(title = 'Terrorist Events',

          xaxis = dict(title = 'Year'),

          yaxis = dict(title = 'Number of Events'),

          )

fig = dict(data=data, layout=layout)

iplot(fig, filename='styled-line')

    
mapbox_access_token = 'pk.eyJ1Ijoic2hpc2hpcmpha2F0aSIsImEiOiJjajhueW8xaHgxZHoyMndydG94aWRkaGlhIn0.hpXRZtotwg3nlsABMBPuYA'

bombing_df = main_df[(main_df['attacktype1'] == 3) & (main_df['nkill'] >= 1)]

## all bombing attacks with fatality data

bombing_data = []

bombing_data.append(Scattermapbox(

    lat = bombing_df['latitude'],

    lon = bombing_df['longitude'],

    mode = 'markers',

    marker = dict(

            size = bombing_df['nkill'],

            sizemode = 'area'

    ),

    text = bombing_df['nkill']

))

bombing_layout = Layout(

    autosize=True,

    hovermode='closest',

    mapbox=dict(

        accesstoken=mapbox_access_token,

        bearing=0,

        center=dict(

            lat=45,

            lon=45

        ),

        pitch=0,

        zoom=1,

    ),

)



fig = dict(data=bombing_data, layout=bombing_layout)

iplot(fig, filename='Montreal Mapbox')
correlations_df = bombing_df[['iday', 'imonth', 'iyear', 'country', 'region', 'latitude', 'longitude', 'targtype1', 'attacktype1', 'targsubtype1', 'natlty1', 'weaptype1', 'weapsubtype1', 'nwound', 'nkill', 'property', 'related']].copy()

corr = correlations_df.corr()

sns.heatmap(corr, cmap='coolwarm')

plt.show()