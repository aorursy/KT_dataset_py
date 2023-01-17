import pandas as pd

import requests

import plotly.express as px

import matplotlib.pyplot as plt

import plotly.offline as po

po.init_notebook_mode (connected = True)




regions = ['Piemonte', 'Trentino-Alto Adige', 'Lombardia', 'Puglia', 'Basilicata', 

           'Friuli Venezia Giulia', 'Liguria', "Valle d'Aosta", 'Emilia-Romagna',

           'Molise', 'Lazio', 'Veneto', 'Sardegna', 'Sicilia', 'Abruzzo',

           'Calabria', 'Toscana', 'Umbria', 'Campania', 'Marche']



# Create a dataframe with the region names

df = pd.DataFrame(regions, columns=['NOME_REG'])

# For demonstration, create a column with the length of the region's name

df['name_length'] = df['NOME_REG'].str.len()



# Read the geojson data with Italy's regional borders [enter image description here][2]from github

repo_url = 'https://gist.githubusercontent.com/datajournalism-it/48e29e7c87dca7eb1d29/raw/2636aeef92ba0770a073424853f37690064eb0ea/regioni.geojson'

italy_regions_geo = requests.get(repo_url).json()



# Choropleth representing the length of region names

fig = px.choropleth(data_frame=df, 

                    geojson=italy_regions_geo, 

                    locations='NOME_REG', # name of dataframe column

                    featureidkey='properties.NOME_REG',  # path to field in GeoJSON feature object with which to match the values passed in to locations

                    color='name_length',

                    color_continuous_scale="Magma",

                    scope="europe",

                   )

fig.update_geos(showcountries=False, showcoastlines=False, showland=False, fitbounds="locations")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
italy_regions_geo['features'][0]
#alternative geojson for austria

#http://data.opendataportal.at/dataset/geojson-daten-osterreich/resource/1a8718bb-18b1-47e1-b6a5-2af5190e087e'

jsonurl = 'https://github.com/ginseng666/GeoJSON-TopoJSON-Austria/raw/master/2017/simplified-95/bezirke_95_geo.json'
austria_regions_geo = requests.get(jsonurl).json()
austria_regions_geo['features'][0]['properties'].items()
#mr of districts in geojson

dist = len(austria_regions_geo['features'])

isolist = []

namelist = []

for i in range(dist):

    isolist.append(austria_regions_geo['features'][i]['properties']['iso'])

    namelist.append(austria_regions_geo['features'][i]['properties']['name'])

df_dist = pd.DataFrame([isolist, namelist]).transpose()

df_dist.rename(columns={0:'iso', 1:'name'}, inplace=True)
# make a feature to plot

df_dist['namelen'] = df_dist.name.str.len()

df_dist.head()
px.histogram(df_dist, x='namelen')
tsbez = pd.read_csv('https://covid19-dashboard.ages.at/data/CovidFaelle_Timeline_GKZ.csv', sep=';', decimal=',')
tsbez.GKZ = tsbez.GKZ.astype(str)

tsbez['Date'] = pd.to_datetime(tsbez.Time, dayfirst=True)

tsbez_curr = tsbez.loc[tsbez.Date==tsbez.Date.max()]
tsbez_curr.head()
df_merged = pd.merge(df_dist, tsbez_curr, left_on='iso', right_on='GKZ', how='left')
df_merged.columns
fig = px.choropleth(data_frame=df_merged, 

                    geojson=austria_regions_geo, 

                    locations='iso', # name of dataframe column

                    featureidkey='properties.iso',  # path to field in GeoJSON feature object with which to match the values passed in to locations

                    hover_name='Bezirk',

                    color='SiebenTageInzidenzFaelle',

                    color_continuous_scale="balance",

                    scope="europe",

                   )

fig.update_geos(showcountries=False, showcoastlines=False, showland=False, fitbounds="locations")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
# px with mapbox
fig = px.choropleth_mapbox(data_frame=df_merged, 

                    geojson=austria_regions_geo, 

                    locations='iso', # name of dataframe column

                    featureidkey='properties.iso',  # path to field in GeoJSON feature object with which to match the values passed in to locations

                    center={"lat": 47.4911, "lon": 14.1785},

                    mapbox_style="carto-positron", 

                    zoom=6,

                    hover_name='Bezirk',

                    color='SiebenTageInzidenzFaelle',

                    color_continuous_scale="balance",

                    opacity = 0.5

                   )

fig.update_geos(showcountries=False, showcoastlines=False, showland=False, fitbounds="locations")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
