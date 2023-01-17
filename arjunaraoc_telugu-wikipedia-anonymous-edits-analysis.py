# Telugu wikipedia anonymous edits analysis
# inputs
# --Recent changes of anonymous edits(IP address users) with a limit on the number of days and also max entries (default 256) through wikipedia API
# --Geolite2-city database to do reverse lookup and find out country, subdivision
# --India states features with country and subdivision codes (single digit and two digit precision data provided)
# --  two digit precision data used for better appearance when zoomed
# --India official boundary Geojson to be compliant with Indian laws for public publishing
# --ISO-3 codes and names database for use in overlays etc(optional)
# Outputs
# --State boundaries filled as per the number of edits from that state (choropleth map)

# Version History
# V25: got the plot.ly bar chart working by not using FigureWidget
# V23: Figurewidget plots not being shown in online viewer probably because of somebug
# V22: Added top 10 cities as bar chart 
# V21: Moved to geoip2 package and made it work

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import requests
import json
import os
import folium
#URL for query of latest 1024 anonymous edits on te.wikipedia.org 
# https://te.wikipedia.org/w/api.php?action=query&format=json&list=recentchanges&rcprop=title%7Cids%7Csizes%7Cflags%7Cuser&rcshow=anon&rclimit=256
urls="https://te.wikipedia.org/w/api.php?action=query&format=json&list=recentchanges&rcprop=title%7Cids%7Csizes%7Cflags%7Cuser&rcshow=anon&rclimit=1024"
r = requests.get(urls)
data=r.json()
rc=pd.DataFrame(data['query']['recentchanges'])
# Group by User IP
rcg=rc.groupby('user')
# Count edits by User IP
rcip=rcg.size()

# Do IP lookup and get state codes
import geoip2.database
# This creates a Reader object. You should use the same object
# across multiple requests as creation of it is expensive.
reader = geoip2.database.Reader('../input/geolite2city/GeoLite2-City.mmdb')
#db=open_database("../input/geolite2city/GeoLite2-City.mmdb")
rccountry=[]
rcstate=[]
rccity=[]
for x in rcip.index:
    response = reader.city(x)
    rccountry.append(response.country.iso_code)
    rcstate.append(response.subdivisions.most_specific.iso_code)
    rccity.append(response.city.name)
rctab=pd.DataFrame({'country':rccountry,'state':rcstate,'city':rccity})
rctab=rctab.dropna()
rctab=rctab[rctab.country=="IN"]
rctab['country_state']=rctab['country']+"."+rctab['state']
rcdf=rctab['country_state'].value_counts().rename_axis('country_state').reset_index(name='edits')




## construct  and display chorpleth map of anonymous edits on Telugu Wikipedia
india_geo = os.path.join('../input/india-states-precision-1', 'india-states-p1.geojson')
geo_json_data = json.load(open(india_geo))
india_off= os.path.join('../input/india-official-boundary','india-composite.json')
m = folium.Map(location=[21.14, 79.08], zoom_start=4)
# Provide a style_function that color all states green but Alabama.
style_function = lambda x: {'fillColor': '#0000ff','fillOpacity':'0'}


folium.features.GeoJson(india_off,style_function=style_function).add_to(m)
# properties.HASC_1 of geo_data and 'state' of data are India admin regions level1 codes like 
# IN.AP for Andhrapradesh state of India
folium.Choropleth(
    geo_data=geo_json_data,
    name='choropleth',
    data=rcdf,
    columns=['country_state', 'edits'],
    key_on='feature.properties.HASC_1',
    fill_color='YlGn',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Anonymous edits on Telugu Wikipedia (%)'
).add_to(m)

folium.LayerControl().add_to(m)
m
# Draw a simple bar chart of number of edits by city
rctab['state_city']=rctab['state']+"_"+rctab['city']
rcdfcitytop10=rctab['state_city'].value_counts().rename_axis('state_city').reset_index(name='edits')[1:10]                                                                                
# import plotly
import plotly.plotly as py
import plotly.graph_objs as go

# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

# sepcify that we want a scatter plot with, with date on the x axis and meet on the y axis
data = [go.Bar(x=rcdfcitytop10.state_city, y=rcdfcitytop10.edits)]

# specify the layout of our figure
layout = dict(title = "Anonymous edits vs Indian state, city on Telugu Wikipedia",
              xaxis= dict(title= 'State_City',zeroline= False))

# create and show our figure
fig = dict(data = data, layout = layout)
iplot(fig)