# libraries

import numpy as np

import pandas as pd

import os

import copy

import matplotlib.pyplot as plt

%matplotlib inline

from tqdm import tqdm_notebook

pd.options.display.precision = 15

from collections import defaultdict

import time

from collections import Counter

import datetime

import gc

import seaborn as sns

import shap

from IPython.display import HTML

import json

import networkx as nx

from typing import List

import requests

import os

import time

import datetime

import json

from itertools import product

pd.set_option('max_rows', 500)

import re



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import plotly.express as px



import sys

from itertools import chain

plt.style.use('seaborn')
# read the json with the data

with open('/kaggle/input/coronavirus-infection-in-finland/v2', 'r') as f:

    data = json.load(f)

    

# get geo data for plotting

r = requests.get(url='https://raw.githubusercontent.com/deldersveld/topojson/master/countries/finland/finland-regions.json')

d = r.json()
data.keys()
data['confirmed'][0]
print(f"{len(data['confirmed'])} confirmed cases right now")
data['deaths'][0]
print(f"{len(data['deaths'])} deaths right now")
data['recovered'][0]
print(f"{data['recovered'][-1]['id']} persons recovered")
df = pd.DataFrame(data['confirmed'])

df['date'] = pd.to_datetime(df['date']).dt.date
df['healthCareDistrict'].value_counts().plot(kind='barh')

plt.title('Number of confirmed cases per district');
# df['infectionSource'].value_counts().plot(kind='barh')

# plt.title('Number of confirmed cases by infection country');

pd.crosstab(df['healthCareDistrict'], df['infectionSourceCountry'].astype(str))
def rel2abs(arc, scale=None, translate=None):

    """Yields absolute coordinate tuples from a delta-encoded arc.

    If either the scale or translate parameter evaluate to False, yield the

    arc coordinates with no transformation."""

    if scale and translate:

        a, b = 0, 0

        for ax, bx in arc:

            a += ax

            b += bx

            yield scale[0]*a + translate[0], scale[1]*b + translate[1]

    else:

        for x, y in arc:

            yield x, y



def coordinates(arcs, topology_arcs, scale=None, translate=None):

    """Return GeoJSON coordinates for the sequence(s) of arcs.

    

    The arcs parameter may be a sequence of ints, each the index of a

    coordinate sequence within topology_arcs

    within the entire topology -- describing a line string, a sequence of 

    such sequences -- describing a polygon, or a sequence of polygon arcs.

    

    The topology_arcs parameter is a list of the shared, absolute or

    delta-encoded arcs in the dataset.

    The scale and translate parameters are used to convert from delta-encoded

    to absolute coordinates. They are 2-tuples and are usually provided by

    a TopoJSON dataset. 

    """

    if isinstance(arcs[0], int):

        coords = [

            list(

                rel2abs(

                    topology_arcs[arc if arc >= 0 else ~arc],

                    scale, 

                    translate )

                 )[::arc >= 0 or -1][i > 0:] \

            for i, arc in enumerate(arcs) ]

        return list(chain.from_iterable(coords))

    elif isinstance(arcs[0], (list, tuple)):

        return list(

            coordinates(arc, topology_arcs, scale, translate) for arc in arcs)

    else:

        raise ValueError("Invalid input %s", arcs)



def geometry(obj, topology_arcs, scale=None, translate=None):

    """Converts a topology object to a geometry object.

    

    The topology object is a dict with 'type' and 'arcs' items, such as

    {'type': "LineString", 'arcs': [0, 1, 2]}.

    See the coordinates() function for a description of the other three

    parameters.

    """

    return {

        "type": obj['type'], 

        "coordinates": coordinates(

            obj['arcs'], topology_arcs, scale, translate )}



from shapely.geometry import asShape



topojson_path = sys.argv[1]

geojson_path = sys.argv[2]



topology = d



# file can be renamed, the first 'object' is more reliable

layername = list(topology['objects'].keys())[0]  



features = topology['objects'][layername]['geometries']

scale = topology['transform']['scale']

trans = topology['transform']['translate']



fc = {'type': "FeatureCollection", 'features': []}



for id, tf in enumerate(features):

    f = {'id': id, 'type': "Feature"}

    f['properties'] = tf['properties'].copy()



    geommap = geometry(tf, topology['arcs'], scale, trans)

    geom = asShape(geommap).buffer(0)

    assert geom.is_valid

    f['geometry'] = geom.__geo_interface__



    fc['features'].append(f) 
district_id = {j['properties']['VARNAME_2'].split('|')[0]: j['id'] for j in fc['features']}

district_id['HUS'] = district_id['Uusimaa']

district_id['Vaasa'] = district_id['Österbotten']

del district_id['Uusimaa']

del district_id['Österbotten']



district_mapping = {'Etelä-Karjala': 0,

 'Etelä-Pohjanmaa': 1,

 'HUS': 2,

 'Kanta-Häme': 3,

 'Keski-Suomi': 4,

 'Lappi': 5,

 'Pirkanmaa': 6,

 'Pohjois-Karjala': 7,

 'Pohjois-Pohjanmaa': 8,

 'Pohjois-Savo': 9,

 'Satakunta': 10,

 'Varsinais-Suomi': 11}



df = df.sort_values('date')

df.loc[df['healthCareDistrict'].isnull(), 'healthCareDistrict'] = 'HUS'

df['total'] = df.sort_values('date').groupby('healthCareDistrict').cumcount()

df.loc[df['healthCareDistrict'].isnull(), 'healthCareDistrict'] = 'HUS'

df = df.groupby(['date', 'healthCareDistrict']).size().to_frame().reset_index()



zero_regions = [k for k in district_id.keys() if k not in df['healthCareDistrict'].unique()]

min_date = df['date'].min()

for reg in zero_regions:

    df.loc[len(df)] = list((df['date'].min(), reg, 0))



df = pd.melt(df.pivot(index='date', columns='healthCareDistrict', values=0).fillna(0).reset_index(), id_vars=['date'])

df = df.sort_values(['date', 'healthCareDistrict'])

df.loc[df['healthCareDistrict'] == 'Länsi-Pohja', 'healthCareDistrict'] = 'Lappi'

df.loc[df['healthCareDistrict'] == 'Ahvenanmaa', 'healthCareDistrict'] = 'Varsinais-Suomi'

df.loc[df['healthCareDistrict'] == '', 'healthCareDistrict'] = 'HUS'

df.loc[df['healthCareDistrict'].isnull(), 'healthCareDistrict'] = 'HUS'

df.loc[df['healthCareDistrict'] == 'Itä-Savo', 'healthCareDistrict'] = 'Pohjois-Savo'



df['id'] = df['healthCareDistrict'].apply(lambda x: district_id[x])

df['date'] = df['date'].astype(str)

df['total_confirmed'] = df.sort_values('date').groupby('healthCareDistrict')['value'].cumsum()
fig = px.choropleth(df,

                    geojson=fc,

                    locations='id',

                    animation_frame='date',

                    color_continuous_scale="OrRd",

                    hover_name='healthCareDistrict',

                    range_color=(0, df['total_confirmed'].max()),

                    color='total_confirmed')



fig.update_geos(fitbounds="locations", visible=False)

fig.update_geos(projection_type="orthographic")

fig.update_layout(height=600, margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
px.line(df, x='date' , y='value' , color='healthCareDistrict', title='Number of new confirmed cases by district')
px.bar(df.groupby('date')['value'].sum().reset_index(), x='date' , y='value', title='Number of new confirmed cases by date')
df_deaths = pd.DataFrame(data['deaths'])

df_deaths['date'] = pd.to_datetime(df_deaths['date']).dt.date

df_deaths = df_deaths.sort_values('date')

df_deaths.loc[df_deaths['healthCareDistrict'].isnull(), 'healthCareDistrict'] = 'HUS'

df_deaths['total'] = df_deaths.sort_values('date').groupby('healthCareDistrict').cumcount()



df_deaths = df_deaths.groupby(['date', 'healthCareDistrict']).size().to_frame().reset_index()



zero_regions = [k for k in district_id.keys() if k not in df_deaths['healthCareDistrict'].unique()]

min_date = df_deaths['date'].min()

for reg in zero_regions:

    df_deaths.loc[len(df_deaths)] = list((df_deaths['date'].min(), reg, 0))



df_deaths = pd.melt(df_deaths.pivot(index='date', columns='healthCareDistrict', values=0).fillna(0).reset_index(), id_vars=['date'])

df_deaths = df_deaths.sort_values(['date', 'healthCareDistrict'])

df_deaths.loc[df_deaths['healthCareDistrict'] == 'Länsi-Pohja', 'healthCareDistrict'] = 'Lappi'

df_deaths.loc[df_deaths['healthCareDistrict'] == 'Ahvenanmaa', 'healthCareDistrict'] = 'Varsinais-Suomi'

df_deaths.loc[df_deaths['healthCareDistrict'] == '', 'healthCareDistrict'] = 'HUS'

df_deaths.loc[df_deaths['healthCareDistrict'].isnull(), 'healthCareDistrict'] = 'HUS'

df_deaths.loc[df_deaths['healthCareDistrict'] == 'Itä-Savo', 'healthCareDistrict'] = 'Pohjois-Savo'



df_deaths['id'] = df_deaths['healthCareDistrict'].apply(lambda x: district_id[x])

df_deaths['date'] = df_deaths['date'].astype(str)

df_deaths['total_confirmed'] = df_deaths.sort_values('date').groupby('healthCareDistrict')['value'].cumsum()
px.bar(df_deaths.groupby('date')['value'].sum().reset_index(), x='date' , y='value', title='Number of deaths by date')
# df1 = pd.DataFrame(data['confirmed']).groupby(['infectionSourceCountry', 'healthCareDistrict']).size().reset_index()



# G = nx.from_pandas_edgelist(df1, 'infectionSourceCountry', 'healthCareDistrict', [0])

# colors = []

# for node in G:

#     if node in df1["infectionSourceCountry"].unique():

#         colors.append("red")

#     else:

#         colors.append("lightgreen")

        

# nx.draw(nx.from_pandas_edgelist(df1, 'infectionSourceCountry', 'healthCareDistrict', [0]), with_labels=True, node_color=colors)