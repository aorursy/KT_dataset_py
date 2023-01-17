import numpy as np # linear algebra
from numpy import log10, ceil, ones
from numpy.linalg import inv 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # prettier graphs
import matplotlib.pyplot as plt # need dis too
%matplotlib inline 
from IPython.display import HTML # for da youtube memes
import itertools # let's me iterate stuff
from datetime import datetime # to work with dates
import geopandas as gpd
from fuzzywuzzy import process
from shapely.geometry import Point, Polygon
import shapely.speedups
shapely.speedups.enable()
import fiona 
from time import gmtime, strftime
from shapely.ops import cascaded_union
import gc
import folium # leaflet.js py map
from folium import plugins

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

sns.set_style('darkgrid') # looks cool, man
import os

df_donations = pd.read_csv("../input/io/Donations.csv")
df_donors = pd.read_csv("../input/io/Donors.csv", low_memory=False)
df_projects = pd.read_csv("../input/io/Projects.csv", error_bad_lines=False)
df_resources = pd.read_csv("../input/io/Resources.csv", error_bad_lines=False)
df_schools = pd.read_csv("../input/io/Schools.csv", error_bad_lines=False)
df_teachers = pd.read_csv("../input/io/Teachers.csv", error_bad_lines=False)

df_population = pd.read_csv("../input/population-by-state/population.csv")
df_st_abbrev = pd.read_csv("../input/state-abbreviations/state_abbrev.csv")

df_schools.rename(columns={'School State': 'state', 'School City': 'city'}, inplace=True)
df_donors.rename(columns={'Donor State': 'state', 'Donor City': 'city'}, inplace=True)
df_donors[df_donors['city'] == 'Seattle'][['city', 'state']].groupby(['city', 'state']).size().reset_index().rename(columns={0: 'count'}).sort_values('count', ascending=False).head(15)
df1 = df_schools[~df_schools['city'].isnull()][['city', 'state']]
df2 = df_donors[(df_donors['state'] != 'other') & (~df_donors['city'].isnull())][['city', 'state']]
# all existing seemingly valid city state pairs
frames = [df1, df2]
df_city_state_map = pd.concat(frames)
df_city_state_map.rename(columns={'state': 'map_state', 'city': 'map_city'}, inplace=True)

# count, keep the city/state pair with largest count as map
df_city_state_map = df_city_state_map.groupby(['map_city', 'map_state']).size().reset_index().rename(columns={0: 'count'})
df_city_state_map['rank'] = df_city_state_map.groupby(['map_city'])['count'].rank(ascending=False)
df_city_state_map = df_city_state_map[df_city_state_map['rank'] == 1]

# fix the data
df_donors = df_donors.merge(df_city_state_map[['map_city', 'map_state']], how='left', left_on='city', right_on='map_city')
df_donors['state'] = np.where((df_donors['state'] == 'other') & (~df_donors['city'].isnull()), df_donors['map_state'], df_donors['state'])
df_donors.drop(columns=['map_city', 'map_state'], inplace=True)

# show count now
df_donors[df_donors['city'] == 'Seattle'][['city', 'state']].groupby(['city', 'state']).size().reset_index().rename(columns={0: 'count'}).sort_values('count', ascending=False).head(5)
df_schools = df_schools.merge(df_st_abbrev[['State', 'Abbreviation']], how='left', left_on='state', right_on='State').drop(columns=['State'])
df_schools.rename(columns={'Abbreviation': 'st'}, inplace=True)
df_donors = df_donors.merge(df_st_abbrev[['State', 'Abbreviation']], how='left', left_on='state', right_on='State').drop(columns=['State'])
df_donors.rename(columns={'Abbreviation': 'st'}, inplace=True)
df_state = df_donors.groupby(['st']).size().reset_index().rename(columns={0: 'count'})

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]


data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = df_state['st'],
        z = df_state['count'],
        locationmode = 'USA-states',
        #text = df['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = 'count of donors')
        ) ]

layout = dict(
        title = 'Donors by State',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
py.iplot( fig, filename='d3-cloropleth-map' )
df_state = df_state.merge(df_population, how='left', left_on='st', right_on='State').drop(columns='State')
df_state['donors_per_pop'] = df_state['count'] / df_state['Population']

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]


data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = df_state['st'],
        z = df_state['donors_per_pop'],
        locationmode = 'USA-states',
        #text = df['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = 'registered donors per population')
        ) ]

layout = dict(
        title = 'Donors by State Per Capita',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
py.iplot( fig, filename='d3-cloropleth-map' )
df_donations = df_donations.merge(df_donors[['Donor ID', 'state', 'st', 'Donor Is Teacher']], how='left', on='Donor ID').drop(columns=['Donor ID'])
df_donations.rename(columns={'state': 'Donor State', 'st': 'Donor ST'}, inplace=True)
df_state = df_donations.groupby(['Donor ST'])['Donation Amount'].median().reset_index()

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]


data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = df_state['Donor ST'],
        z = df_state['Donation Amount'],
        locationmode = 'USA-states',
        #text = df['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = 'Median Donations')
        ) ]

layout = dict(
        title = 'median donation FROM state',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
py.iplot( fig, filename='d3-cloropleth-map' )
df_state = df_donations.groupby(['Donor ST'])['Donation Amount'].mean().reset_index()

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]


data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = df_state['Donor ST'],
        z = df_state['Donation Amount'],
        locationmode = 'USA-states',
        #text = df['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = 'Mean Donations')
        ) ]

layout = dict(
        title = 'mean donation FROM state',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
py.iplot( fig, filename='d3-cloropleth-map' )
df_state = df_donations.merge(df_projects[['Project ID', 'School ID']], how='left', on='Project ID')
df_state = df_state.merge(df_schools[['School ID', 'state', 'st']], how='left', on='School ID')
df_state.rename(columns={'state': 'School State', 'st': 'School ST'}, inplace=True)
df_display = df_state.groupby(['School ST'])['Donation Amount'].median().reset_index()

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]


data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = df_display['School ST'],
        z = df_display['Donation Amount'],
        locationmode = 'USA-states',
        #text = df['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = 'Median Donations')
        ) ]

layout = dict(
        title = 'median donation TO state',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
py.iplot( fig, filename='d3-cloropleth-map' )
df_state['donation_within'] = np.where(df_state['Donor ST'] == df_state['School ST'], 'Yes', 'No')
df_display = df_state[~df_state['Donor ST'].isnull()]['donation_within'].value_counts().to_frame().reset_index()
df_display.rename(columns={'donation_within': 'count'}, inplace=True)
df_display.rename(columns={'index': 'donation_within'}, inplace=True)

trace = go.Pie(labels=df_display['donation_within'], values=df_display['count'], marker=dict(colors=['#75e575', '#ea7c96']))

py.iplot([trace], filename='basic_pie_chart')
df_display = df_state[~df_state['Donor ST'].isnull()].groupby(['Donor ST', 'donation_within']).size().reset_index().rename(columns={0: 'count'})
df_display = df_display.pivot(index='Donor ST', columns='donation_within', values='count').reset_index()
df_display['percent_within_state'] = df_display['Yes'] / (df_display['No'] + df_display['Yes'])

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]


data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = df_display['Donor ST'],
        z = df_display['percent_within_state'],
        locationmode = 'USA-states',
        #text = df['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = 'Same State Donations')
        ) ]

layout = dict(
        title = 'Percent of Donations Kept Within State',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
py.iplot( fig, filename='d3-cloropleth-map' )
df_pie = df_donations.groupby('Donor Is Teacher').size().to_frame().reset_index().rename(columns={0: 'count'})

trace = go.Pie(labels=df_pie['Donor Is Teacher'], values=df_pie['count'], marker=dict(colors=['#ea7c96', '#75e575']))

py.iplot([trace], filename='basic_pie_chart')
df_pie = df_donations.groupby('Donation Included Optional Donation').size().to_frame().reset_index().rename(columns={0: 'count'})

trace = go.Pie(labels=df_pie['Donation Included Optional Donation'], values=df_pie['count'], marker=dict(colors=['#ea7c96', '#75e575']))

py.iplot([trace], filename='basic_pie_chart')
df_schools.groupby('School Metro Type')['School Percentage Free Lunch'].describe().reset_index().sort_values('50%')
df_schools.sort_values('School Percentage Free Lunch', ascending=False).head()
df_schools[df_schools['city'] == 'Chicago']['School Percentage Free Lunch'].describe().to_frame()
df_zips = pd.read_csv("../input/us-zip-codes-with-lat-and-long/zip_lat_long.csv")
gdf_areas = gpd.read_file('../input/chicago-community-areas-geojson/chicago-community-areas.geojson')

epsg = '32616'

gdf_schools = df_schools.merge(df_zips, how='left', left_on='School Zip', right_on='ZIP')
gdf_schools['geometry'] = gdf_schools.apply(lambda row: Point(row['LNG'], row['LAT']), axis=1)
gdf_schools = gpd.GeoDataFrame(gdf_schools, geometry='geometry')
gdf_schools.drop(columns=['ZIP', 'LAT', 'LNG'], inplace=True)
gdf_schools.crs = {'init': epsg}

gdf_chi_schools = gdf_schools[gdf_schools['city'] == 'Chicago']
gdf_chi_schools['community'] = np.NaN
gdf_chi_schools['r_map'] = np.NaN
### POINTS IN POLYGONS
for i in range(0, len(gdf_areas)):
    print('i is: ' + str(i) + ' at ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    gdf_chi_schools['r_map'] = gdf_chi_schools.within(gdf_areas['geometry'][i])
    gdf_chi_schools['community'] = np.where(gdf_chi_schools['r_map'], gdf_areas['community'][i], gdf_chi_schools['community'])
df_temp = gdf_chi_schools.groupby('community')['School Percentage Free Lunch'].mean().to_frame().reset_index()
df_temp.rename(columns={'School Percentage Free Lunch': 'free_lunch_mean_zip'}, inplace=True)
gdf_areas = gdf_areas.merge(df_temp, how='left', on='community')
# create Chicago map
CHICAGO_COORDINATES = [41.85, -87.68]
map_attributions = ('&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a> '
        'contributors, &copy; <a href="http://cartodb.com/attributions">CartoDB</a>')
community_map = folium.Map(location=CHICAGO_COORDINATES, 
                                 attr=map_attributions,
                         #        tiles=None, #'Cartodb Positron', #'OpenStreetMap',
                                 zoom_start=10, min_zoom=10,
                                 control_scale=True)

geojson = gdf_areas.to_json()
geojson = gdf_areas[~gdf_areas['free_lunch_mean_zip'].isnull()].to_json()

# map Chicago communities crime
community_map.choropleth(
    #geo_data='../input/chicago-community-areas-geojson/chicago-community-areas.geojson',
    geo_data=geojson,
    data=gdf_areas[~gdf_areas['free_lunch_mean_zip'].isnull()],
    columns=['community', 'free_lunch_mean_zip'],
    key_on='feature.properties.community',
    line_opacity=0.3,
    fill_opacity=0.5,
    fill_color='YlOrRd',
    #legend_name='Chicago Crime by Community (2001-2017)', 
    highlight=True, 
    #threshold_scale=[30, 41, 63, 73, 84, 95],
    smooth_factor=2
)

# add fullscreen toggle
#plugins.Fullscreen(
#    position='topright',
#    title='full screen',
#    title_cancel='exit full screen',
#    force_separate_button=True).add_to(community_crime_map)

# add base map tile options
folium.TileLayer('OpenStreetMap').add_to(community_map)
#folium.TileLayer('stamentoner').add_to(community_map)
folium.TileLayer('Cartodb Positron').add_to(community_map)
folium.LayerControl().add_to(community_map)

# show map
community_map










Blues = plt.get_cmap('Blues')

regions = gdf_areas[~gdf_areas['free_lunch_mean_zip'].isnull()]['community']

fig, ax = plt.subplots(1, figsize=(12,12))

for r in regions:
    gdf_areas[gdf_areas['community'] == r].plot(ax=ax, color=Blues(gdf_areas[gdf_areas['community'] == r]['free_lunch_mean_zip'] / 100/1.35))
    #gdf_areas[gdf_areas['community'] == r].plot(ax=ax, color='powderblue')

gdf_schools[(gdf_schools['city'] == 'Chicago')].plot(ax=ax, markersize=10, color='red')

for i, point in gdf_areas.centroid.iteritems():
    reg_n = gdf_areas.iloc[i]['community']
    reg_n = gdf_areas.loc[i, 'community']
    ax.text(s=reg_n, x=point.x, y=point.y, fontsize='small')
    

ax.set_title('Chicago Neighborhoods; Darker Shade = Higher Percentage Free Lunch Qualification')
ax.legend(loc='upper left', frameon=True)
leg = ax.get_legend()
#new_title = 'Partner ID'
#leg.set_title(new_title)

plt.show()








df_projects.head()
df_resources.head()
df_schools.head()
df_teachers.head()
df_donations.head()
df_schools.head()



















