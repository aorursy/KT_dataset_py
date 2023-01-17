import plotly.express as px
import geopandas as gpd

import plotly.graph_objects as go
from shapely.geometry import LineString, MultiLineString

import numpy as np
import json

import pandas as pd

f = open("../input/geojscv/geojsdata.txt").read()
f = f.replace("\'", "\"") 

geojsdata = json.loads(f)


nsw_covidinfo = pd.read_csv("../input/new-covidinfo/nsw_covidinfo.csv")
nsw_cases = nsw_covidinfo.drop(nsw_covidinfo[nsw_covidinfo.num_of_cases == 0].index)

nsw_cases = nsw_cases.groupby(['postcode','num_of_tests','num_of_cases','latitude','longitude'])['locality'].apply(','.join).reset_index()
nsw_cases = nsw_cases.sort_values('num_of_cases', ascending=False)
nsw_cases = nsw_cases.head(8).reset_index()
nsw_cases = nsw_cases.rename(columns={'locality': 'suburb'})
nsw_cases.style.bar(subset=['num_of_tests', 'num_of_cases'], align='mid', color=['#d65f5f', '#5fba7d'])
token = '_'

locations = []
i = 1 
for i in range(312):
    locations.append(str(i).zfill(2))

name = []
i = 1 
for i in range(312):
    name.append(geojsdata["features"][i]['properties']['name'])


data = {'location': locations, 'rand':np.random.randint(313,size = 312),'name':name}
dfdata = pd.DataFrame(data)
# print(dfdata.head())



shape = go.Choroplethmapbox(geojson=geojsdata, locations=dfdata['location'], z=dfdata['rand'],
                            text=dfdata['name'], hoverinfo='text',
                                    marker_opacity=0.5, marker_line_width=0)



lats = nsw_cases["latitude"].tolist()
lons = nsw_cases["longitude"].tolist()
text = nsw_cases['num_of_cases'].tolist()
name = nsw_cases['suburb'].tolist()


point = go.Scattermapbox(lat=lats,
        lon=lons,
        mode='markers',
        text=text,   
        name='cases',
        hoverinfo='text',                 
        below='',                 
        marker=dict( size=10, color ='yellow'),
                        showlegend=False)

layout = go.Layout( title_x =0.5, width=750, height=700,
                   mapbox = dict(center= dict(lat=-33.865143, lon=150.809900),            
                                 accesstoken= token,
                                 zoom=7,
                                style="dark"))
fig=go.Figure(data=[shape, point], layout =layout)
fig.update_layout(coloraxis = {'showscale':False})




fig.show()