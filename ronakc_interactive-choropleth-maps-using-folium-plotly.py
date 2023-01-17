import pandas as pd

import folium

import branca.colormap as cm

import geopandas as gpd

from folium.plugins import Search

import plotly

import branca

import random

import requests

import warnings; warnings.filterwarnings('ignore')
print("geopandas version", gpd.__version__)

print("folium    version", folium.__version__)

print("branca    version", branca.__version__)

print("plotly    version", plotly.__version__)
url = 'https://raw.githubusercontent.com/ronakchittora/Python/master/data_india_shapefile_stanford/stanford-sh819zz8121-geojson.json'

districts = gpd.read_file(url, driver='GeoJSON')

districts.crs = "EPSG:4326"
print(districts.shape, "\n")

districts.head(2)
districts = districts[['id', 'coc', 'nam', 'laa', 'geometry']]
# Getting the state boundries from file ('dissolve' same as 'groupby' in pandas, grouping on state names)

states = districts.dissolve(by='nam')

states.reset_index(inplace=True)

states.head(2)
# We may have multiple polygons (multiple records) for each district in our dataset and 

# so we will first get the list of unique districts and 

# we will create a pandas dataframe with id, District and State fields



df_population = pd.DataFrame.from_dict({'id':districts['id'].values.tolist()

                                        , 'District':districts['laa'].values.tolist()

                                        , 'State':districts['nam'].values.tolist()})



df_temp = df_population.drop_duplicates(subset=['State', 'District'], keep='first')

df_temp.drop(columns=['id'], inplace=True)

print('Distinct districts in data %d' %(df_temp.shape[0]))



# now we have a dataframe with unique districts by state, we will generate a random population for each of them

randomlist = []

for i in range(0,df_temp.shape[0]):

  n = random.randint(10000,1000000)

  randomlist.append(n)



df_temp['Random Population'] = randomlist





# Joining with df_population to get population values corresponding to each district

df_population = df_population.merge(df_temp, how='left', on=['State','District'])

print(df_population.info(), "\n")





# Also add population values to geopandas dataframe -  districts

districts = districts.merge(df_population[['id', 'Random Population']], how='inner', on='id')

print("\n", districts.shape)
districts[districts['laa']=='MUMBAI SUBURBAN'].head(3)
# Getting the coordinates to center the map

x_map=states.centroid.x.mean()

y_map=states.centroid.y.mean()

print(x_map,y_map)
# getting the color bar

colormap = branca.colormap.LinearColormap(

    colors=['#39C694','#88D9C1','#EAE0C7','#7AA7E1', '#5B83D8'],

    index=districts['Random Population'].quantile([0.2,0.4,0.6,0.8]),

    vmin=districts['Random Population'].min(),

    vmax=districts['Random Population'].max()

)



colormap.caption="Population of district"



colormap
# Creating a map centered on coordinates we got before

m = folium.Map(location=[y_map, x_map], zoom_start=4, tiles=None)

folium.TileLayer('CartoDB positron',name="Light Map",control=False).add_to(m)



# Plot States

stategeo = folium.GeoJson(states,

                          name='India States',

                          style_function=lambda x: {'color': 'black', 'weight':1, 'fillOpacity':0},

                          ).add_to(m)



# Plot Districts

distgeo = folium.GeoJson(districts,

                          name='India Districts',

                          style_function=lambda x: {'fillColor': colormap(x['properties']['Random Population']), 'color': 'black',

                                                    'weight':0.5, 'fillOpacity':0.5},

                          tooltip=folium.GeoJsonTooltip(fields=['nam', 'laa', 'Random Population'], 

                                            aliases=['State', 'District', 'Population'], 

                                            localize=True)

                         ).add_to(m)



# Add the searchbar for States

statesearch = Search(layer=stategeo, 

                     geom_type='Polygon', 

                     placeholder="Search for a State", 

                     collapsed=False, 

                     search_label='nam',

                     weight=2

                    ).add_to(m)



# Add the searchbar for districts

distsearch = Search(layer=distgeo, 

                    geom_type='Polygon', 

                    placeholder="Search for a District", 

                    collapsed=True, 

                    search_label='laa'

                   ).add_to(m)



# Add layer control

folium.LayerControl().add_to(m)



# Add color bar

colormap.add_to(m)



# Finally show the map

m
data = requests.get(url)

India = data.json()
India['features'][0]
# We will use same random population data which we generated before with Plotly too

df_population.head(2)
import plotly.express as px



fig = px.choropleth(df_population, geojson=India, locations='id', #featureidkey='properties.laa',

                    color='Random Population',

                    center={'lat':y_map, 'lon':x_map},

                    color_continuous_scale=['#39C694','#88D9C1','#EAE0C7','#7AA7E1', '#5B83D8'],

                    range_color=(df_population['Random Population'].min(), df_population['Random Population'].max()),

                    hover_data=['State','District']

                    )



fig.update_geos(fitbounds="locations", visible=False) #visible set to False to not view rest of the world map

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
fig = px.choropleth_mapbox(df_population, geojson=India, locations='id', color='Random Population',

                           color_continuous_scale=['#39C694','#88D9C1','#EAE0C7','#7AA7E1', '#5B83D8'],

                           range_color=(df_population['Random Population'].min(), df_population['Random Population'].max()),

                           mapbox_style="carto-positron",

                           zoom=3, center = {"lat": y_map, "lon": x_map},

                           opacity=0.6,

                           hover_data=['State', 'District']

                          )

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()