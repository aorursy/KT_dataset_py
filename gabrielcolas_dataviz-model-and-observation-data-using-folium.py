import xarray as xr

import numpy as np

import matplotlib.pyplot as plt

import datetime as dt

import pandas as pd
zone = "NW"     #geographic zone (NW or SE)

model = 'arpege' #weather model (arome or arpege)

MODEL = 'ARPEGE' #weather model (AROME or ARPEGE)

level = '2m'      #vertical level (2m, 10m, P_sea_level or PRECIP)

date = dt.datetime(2016, 2, 14,0,0) # Day example 

#parameter name in the file (cf cells below to know the parameter names -> exploration of metadata)

if level == '2m':

    param = 't2m'

elif level == '10m':

    param = 'u10'

elif level == 'PRECIP':

    param = 'tp'

else:

    param = 'msl'
#### Model data 2m

directory = '/kaggle/input/meteonet/' + zone + '_weather_models_2D_parameters_' + str(date.year) + str(date.month).zfill(2) + '/' + str(date.year) + str(date.month).zfill(2) + '/'

fname = directory + f'{MODEL}/{level}/{model}_{level}_{zone}_{date.year}{str(date.month).zfill(2)}{str(date.day).zfill(2)}000000.nc'

data = xr.open_dataset(fname)  

### Model data 10m

level10m='10m'

fname10m = directory + f'{MODEL}/{level10m}/{model}_{level10m}_{zone}_{date.year}{str(date.month).zfill(2)}{str(date.day).zfill(2)}000000.nc'

data10m = xr.open_dataset(fname10m) 

## Model data precip

levelprecip='PRECIP'

fnameprecip = directory + f'{MODEL}/{levelprecip}/{model}_{levelprecip}_{zone}_{date.year}{str(date.month).zfill(2)}{str(date.day).zfill(2)}000000.nc'

dataprecip = xr.open_dataset(fnameprecip)

# Model data pmer

levelpmer='P_sea_level'

fnamepmer = directory + f'{MODEL}/{levelpmer}/{model}_{levelpmer}_{zone}_{date.year}{str(date.month).zfill(2)}{str(date.day).zfill(2)}000000.nc'

datapmer = xr.open_dataset(fnamepmer)



# Ground stations data

year = '2016'

fname = '/kaggle/input/meteonet/'+zone+'_Ground_Stations/'+zone+'_Ground_Stations/'+zone+'_Ground_Stations_'+year+".csv"

df = pd.read_csv(fname,parse_dates=[4],infer_datetime_format=True)
# COLLECT LAT/LON DES STATIONS.

Lat_station=df['lat'].unique()

Lon_station=df['lon'].unique()

# RECUPERER LES COUPLES LAT/LON

df_latlon=df.loc[:,['lat','lon','number_sta']]

LatLonNumbersta=df_latlon.drop_duplicates(['number_sta'])
LatLonNumbersta.shape #262 NumberSta but more lat/lon ??? 302 why

LatLon=LatLonNumbersta.loc[:,['lat','lon']].values

LatLon= np.around(LatLon,decimals=2)
#FILTER UNIQUE STATIONS

Stations= df["number_sta"].unique()
Longitude_model=data["longitude"].values#tolist().round()

Longitude_model= np.around(Longitude_model,decimals=3) # Round because machine calcul
Mid_point_lat=int(data.latitude.values.shape[0]/2)

Mid_point_lon=int(data.longitude.values.shape[0]/2)

Mid_lat=data.latitude.values[Mid_point_lat]

Mid_lon=data.longitude.values[Mid_point_lon]
#FILTER THE NUMBER OF ARROWS

def filter_function_2D(Array,Step_filter_col,Step_filter_ligne):

    dot_col= np.arange(0,Array.shape[1],step=Step_filter_col,dtype=int)

    dot_ligne= np.arange(0,Array.shape[0],step=Step_filter_ligne,dtype=int)

    Array_output= Array[:,dot_col]

    Array_output= Array_output[dot_ligne,:]

    return Array_output
# PREPARE THE DATA

Step=0

X,Y=np.meshgrid(data.longitude.values,data.latitude.values)

U=data10m['u10'].values[Step,:,:]

V=data10m['v10'].values[Step,:,:]



U_filter= filter_function_2D(U,5,2)

V_filter= filter_function_2D(V,5,2)

X_filter= filter_function_2D(X,5,2)

Y_filter= filter_function_2D(Y,5,2)

print(U_filter.shape)



# MAKE QUIVER PLOT

import matplotlib.pyplot as plt

fig1, ax1 = plt.subplots()



agg_filter= np.array([12.,45.,3.])

kw=dict(alpha=0.8,pickradius=15,picker=10,animated=True)#,norm=Normalize)

q=ax1.quiver(X_filter,Y_filter,U_filter,V_filter,**kw,scale=100,width=0.005)

plt.show()



## Export into GeoJson

import mplleaflet

gj= mplleaflet.fig_to_geojson(fig=fig1)

# USE OF COUNTOUR MAP

!pip install --upgrade pip

!pip install geojsoncontour



# TEMPERATURE CHART

import geojsoncontour



fig,ax=plt.subplots()



DATA=data['t2m'].values[0,:,:] - 273.15



contourf= ax.contourf(X, Y,DATA, 8, alpha=1)

#contourf=ax.clabel(contourf,inline=1,fontsize=10)# THIS IS NOTE UNDERSTANT BY THE CODE TRANSCRIPTION





# Convert matplotlib contourf to geojson

geojson = geojsoncontour.contourf_to_geojson(

    contourf=contourf,

    unit='°C',

    min_angle_deg=3.0,

    ndigits=7,

    stroke_width=5,

    fill_opacity=0.1)
# TIME SLIDER WITH THOSE DATA? OR OTHER WAY TO DO THAT WITH 

DATAPRECIP=dataprecip['tp'][0,:,:]

fig,ax=plt.subplots()

CountourPRECIP= ax.contourf(X, Y,DATAPRECIP, levels= [1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.2,2.4,2.5,2.7,3,3.5,4,4.5,4.7,5], alpha=1)



geojsonprecip = geojsoncontour.contourf_to_geojson(

    contourf=CountourPRECIP,

    unit='kg ms-2',

    min_angle_deg=3.0,

    ndigits=7,

    stroke_width=5,

    fill_opacity=1)

#print(geojsonprecip)
DATAPMER= datapmer['msl'][0,:,:]/100

fig,ax=plt.subplots()

CountourPmer= ax.contour(X, Y,DATAPMER,8, alpha=1,colors='black')



geojsonpmer = geojsoncontour.contour_to_geojson(

    contour=CountourPmer,

    unit='hpa',

    min_angle_deg=3.0,

    ndigits=7,

    stroke_width=3)
# MODULE TO DRAW THE CHART INSIDE THE POPUP STATION

!pip install vega

!pip install altair

from altair import Chart

import altair as alt

import json

import folium

# OTHER MODULE

from folium.map import Popup

from folium.plugins import MarkerCluster

from altair import Chart

import folium.plugins

from folium.plugins import MousePosition

from folium.plugins import HeatMap

from branca import colormap

from folium import features
# DUAL MAP WITH m.m1 and m.m2 possibilites to make features groups

m= folium.plugins.DualMap(location=[Mid_lat,Mid_lon],

           tiles='Stamen Terrain',

           zoom_start=5.5)







#### ADD TOOLTIP T2M TO GEOJSON

tooltip = folium.GeoJsonTooltip(

    fields=["title"],

    localize=True,

    sticky=True,

    labels=False,

    style="""

        font-size : 22px;

        background-color: #F0EFEF;

        border: 2px solid black;

        border-radius: 3px;

        box-shadow: 3px;

    """,)

# ADD T2M TEMPERATURE

folium.features.GeoJson(

    geojson,

    style_function=lambda x: {

        'type':      x['geometry']['type'],  

        'color':     x['properties']['stroke'],

        'weight':    x['properties']['stroke-width'],

    },

name= 'T2m',

show=False,

tooltip=tooltip).add_to(m)









#### ADD TOOLTIP PRECIP TO GEOJSON

tooltipprecip = folium.GeoJsonTooltip(

    fields=["title"],

 #   aliases=["T°"],

    localize=True,

    sticky=True,

    labels=False,

    style="""

        font-size : 22px;

        background-color: #F0EFEF;

        border: 2px solid black;

        border-radius: 3px;

        box-shadow: 3px;

    """,)

#ADD PRECIPITATION

folium.features.GeoJson(

    geojsonprecip,

    style_function=lambda x: {

        'type':      x['geometry']['type'],  

        'color':     x['properties']['stroke'],

        'weight':    x['properties']['stroke-width'],

        'fillColor': x['properties']['fill'],

        'fill-opacity':   1,

    },

name= 'PRECIP',

tooltip=tooltipprecip

).add_to(m)







#### ADD TOOLTIP PMER TO GEOJSON

tooltippmer = folium.GeoJsonTooltip(

    fields=["title"],

    aliases=["T°"],

    localize=True,

    sticky=True,

    labels=False,

    style="""

        font-size : 22px;

        background-color: #F0EFEF;

        border: 2px solid black;

        border-radius: 3px;

        box-shadow: 3px;

    """,)

#ADD Pmer

folium.features.GeoJson(

    geojsonpmer,

    style_function=lambda x: {

        'type':      x['geometry']['type'],

        'color':     x['properties']['stroke'],

        'weight':    x['properties']['stroke-width'],

    },

name= 'Pmer',

tooltip=tooltippmer

).add_to(m)









# ADD QUIVER BY USING MPLLEAFLET TRANSLATE TO MPLEATLET UNDERSTANDABLE

WIND_FLOW=folium.FeatureGroup(name='quiver')



for feature in gj['features']:

    if feature['geometry']['type']== 'Point':

        lon_quiver,lat_quiver = feature['geometry']['coordinates'] # Coordinate of the point

        div = feature['properties']['html'] # Property of the ICON to add

        icon_anchor= (feature['properties']['anchor_x'],feature['properties']['anchor_y']) #The coordinates of the “tip” of the icon (relative to its top left corner).

        icon = folium.features.DivIcon(html=div,icon_anchor=icon_anchor)

        marker = folium.Marker([lat_quiver,lon_quiver],icon=icon,opacity=0.7)

        WIND_FLOW.add_child(marker)



WIND_FLOW.add_to(m)







#ADD COORDINATE SHOW AT THE UPPER RIGHT

formatter = "function(num) {return L.Util.formatNum(num, 2);};"

MousePosition(

    position='topright',

    separator=' | ',

    empty_string='NaN',

    lng_first=True,

    num_digits=20,

    prefix='Coordinates:',

    lat_formatter=formatter,

    lng_formatter=formatter,

).add_to(m)









#ADD STATION CLUSTERING

marker_cluster = MarkerCluster(name='Station temperature').add_to(m)



# PREPARE DATA STATION TO LOOP ON 24HOURS ON 2016-02-14

Heures= np.arange(0,24)

TAB_HEURES=[]

for h in Heures:

    TAB_HEURES.append('2016-02-14 '+str(h)+':00:00')

    

# FILTER THE DATA

Heures = df[df.date.isin(TAB_HEURES)]

T2M_Format = Heures["t"].apply(lambda x : x -273.15)

Heures.loc[:,'t'] = T2M_Format



# LOOP TO CUSTOMIZE EVERY STATIONS DATA

for latlon_loop in LatLon:

    

    # DRAW THE CHART INSIDE THE POPUP

    Heures_station= Heures[ ( Heures.lat == latlon_loop[0]) & (Heures.lon == latlon_loop[1])]

    chart=alt.Chart(Heures_station).mark_line().encode(alt.X('date'),alt.Y('t',scale=alt.Scale(domain=(0,25),clamp=True)))

    Trend_json = chart.to_json() #CONVERT TO JSON

    Trend_dict = json.loads(Trend_json) #CONVERT TO DICT

    # ADD POPUP

    popup = folium.Popup()

    folium.VegaLite(Trend_dict, height=50, width=250).add_to(popup)

    folium.Marker(

    location=latlon_loop,

    popup=popup,

    icon=folium.Icon(icon="stats",color='blue')).add_to(marker_cluster)





    

    

    



# ADD CONFIGURATION LAYERS...

folium.raster_layers.TileLayer(tiles='OpenStreetMap').add_to(m)

folium.raster_layers.TileLayer(tiles='stamentoner').add_to(m)

folium.raster_layers.TileLayer(tiles='stamentoner',overlay=True,show=True,control=False).add_to(m.m2)

folium.LayerControl().add_to(m)









m



# SAVE THE OUTPUT MAP

#outfp = "base_map.html"

#m.save(outfp)