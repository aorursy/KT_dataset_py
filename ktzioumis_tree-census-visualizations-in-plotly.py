import pandas as pd

import numpy as np

import plotly as py

import plotly.graph_objects as go

import json
df_1995=pd.read_csv('../input/tree-census/new_york_tree_census_1995.csv')

df_2005=pd.read_csv('../input/tree-census/new_york_tree_census_2005.csv')

df_2015=pd.read_csv('../input/tree-census/new_york_tree_census_2015.csv')

df_trees=pd.read_csv('../input/tree-census/new_york_tree_species.csv')
f = open('../input/nycntamap/NTA_map.geojson')

nta_json = json.load(f)

for i in list(range(0,len(nta_json['features']))):

    json_id=nta_json['features'][i]['properties']['ntacode']

    nta_json['features'][i]['id']=json_id
df_1995.columns
df_2005.columns
df_2015.columns
data=[]

data.append(go.Choroplethmapbox(geojson=nta_json, 

                                    locations=df_1995.nta_2010.value_counts().drop(index='Unknown').index, 

                                    z=df_1995.nta_2010.value_counts().drop(index='Unknown').values,

                                #'Unknown' tree location is by far most common (23k) but is dropped from the map

                                    colorbar=dict(title='1995'),

                                    colorscale="YlGn"

                                    ))

data.append(go.Choroplethmapbox(geojson=nta_json, 

                                    locations=df_2005.nta.value_counts().index, 

                                    z=df_2005.nta.value_counts().values,

                                    colorbar=dict(title='2005'),

                                    colorscale="YlGn"

                                    ))

data.append(go.Choroplethmapbox(geojson=nta_json, 

                                    locations=df_2015.nta.value_counts().index, 

                                    z=df_2015.nta.value_counts().values,

                                    colorbar=dict(title='2015'),

                                    colorscale="YlGn"

                                    ))

data[0]['visible']=False

data[1]['visible']=False

data[2]['visible']=True

layout=go.Layout(mapbox_style="carto-positron",

                  mapbox_zoom=9, mapbox_center = {"lat": 40.7, "lon": -73.86})

#dropdown code from https://plot.ly/~empet/15237/choroplethmapbox-with-dropdown-menu/#/

layout.update(updatemenus=list([

        dict(

            x=-0.05,

            y=1,

            yanchor='top',

            buttons=list([

                dict(

                    args=['visible', [True, False,False]],

                    label='Year: 1995',

                    method='restyle'

                ),

                dict(

                    args=['visible', [False, True, False]],

                    label='Year: 2005',

                    method='restyle'

                ),

                dict(

                    args=['visible', [False,False, True]],

                    label='Year: 2015',

                    method='restyle')]))]))

fig = go.Figure(data=data,layout=layout)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
nta_count_2005=pd.DataFrame(df_2005.nta.value_counts()).reset_index()

nta_count_2015=pd.DataFrame(df_2015.nta.value_counts()).reset_index()

nta_count_diff=nta_count_2015.merge(nta_count_2005, on='index')

nta_count_diff['change']=nta_count_diff['nta_x']-nta_count_diff['nta_y']

nta_count_diff.head()
nta_count_diff['change'].describe()
import matplotlib.pyplot as plt

nta_count_diff['change'].hist(bins=20)

plt.title('Histogram of Tree Count changes 2005-2015 by NTA')

plt.xlabel('Tree Counts')

plt.show()
data=[]

data.append(go.Choroplethmapbox(geojson=nta_json ,

                                    locations=df_2015.nta.value_counts().index, 

                                    z=nta_count_diff.change.values,

                                    colorbar=dict(title=',2015-2005 Change'),

                                    colorscale=[[0, "red"],

                                                [0.15,'yellow'],

                                               [1, "green"]]

                                    ))



layout=go.Layout(mapbox_style="carto-positron",

                  mapbox_zoom=9, mapbox_center = {"lat": 40.7, "lon": -73.86})





fig = go.Figure(data=data,layout=layout)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
temp_df=df_2015.dropna().loc[df_2015.dropna()['spc_common'].str.contains('pine')][['spc_common','latitude','longitude']]

data=[]

for t in temp_df['spc_common'].value_counts().index:

    data.append(go.Scattermapbox(lat=temp_df.loc[df_2015.dropna()['spc_common']==(t)]['latitude'],

                               lon=temp_df.loc[df_2015.dropna()['spc_common']==(t)]['longitude'],

                               mode='markers',

                                 name=t,

                              marker=go.scattermapbox.Marker(size=5)

                      ))

layout=go.Layout(mapbox_style="carto-positron",

                  mapbox_zoom=9, mapbox_center = {"lat": 40.7, "lon": -73.86})

fig = go.Figure(data=data,layout=layout)

fig.update_layout(margin={"r":0,"t":20,"l":0,"b":0})

fig.show()
temp_df=df_2015.dropna().loc[df_2015.dropna()['spc_common'].str.contains('dogwood')][['spc_common','latitude','longitude']]

data=[]

for t in temp_df['spc_common'].value_counts().index:

    data.append(go.Scattermapbox(lat=temp_df.loc[df_2015.dropna()['spc_common']==(t)]['latitude'],

                               lon=temp_df.loc[df_2015.dropna()['spc_common']==(t)]['longitude'],

                               mode='markers',

                                 name=t,

                              marker=go.scattermapbox.Marker(size=5)

                      ))

layout=go.Layout(mapbox_style="carto-positron",

                  mapbox_zoom=9, mapbox_center = {"lat": 40.7, "lon": -73.86})

fig = go.Figure(data=data,layout=layout)

fig.update_layout(margin={"r":0,"t":20,"l":0,"b":0})

fig.show()