# work with maps
#!pip install folium
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # datavizualization
import seaborn as sns # datavizualization
import folium
import json
import requests
import geopandas
import os
from branca.colormap import linear

df = pd.read_csv("../input/apartment-rental-offers-in-germany/immo_data.csv")
# Coordinates of Capital of States
df2 = pd.DataFrame(np.array([[1,"Stuttgart", 48.78, 9.18],
                             [2,"Munich", 48.133333, 11.566667],
                             [3,"Berlin", 52.52, 13.405],
                             [16,"Erfurt",50.983333, 11.033333],
                             [15,"Kiel",54.323333, 10.139444],
                             [14,"Magdeburg",52.133333, 11.616667],
                             [13,"Dresden",51.033333, 13.733333],
                             [12,"Saarbrücken",49.233333, 7],
                             [11,"Mainz",50, 8.266667],
                             [10,"Düsseldorf",51.233333, 6.783333],
                             [9,"Hanover",52.366667, 9.716667],
                             [8,"Schwerin",53.633333, 11.416667],
                             [7,"Wiesbaden",50.0825, 8.24],
                             [6,"Hamburg",53.565278, 10.001389],
                             [5,"Bremen",53.083333, 8.8],
                             [4,"Potsdam",52.4, 13.066667],
                            ]), 
                   columns=['ID_1','city', 'lat', 'lon'])
df['regio1'] = df['regio1'].str.replace('_','-')
datarows_original = df.shape[0]
print("We have {} appartments/rows".format(datarows_original))
df = df.drop(['description',
              'facilities', 
              'picturecount',
              'telekomHybridUploadSpeed', 
              'houseNumber',
              'telekomTvOffer', 
              'telekomUploadSpeed', 
              'streetPlain', 
              'firingTypes', 
              'pricetrend',
              'baseRentRange', 
              'noRoomsRange', 
              'thermalChar',
              'date',
              'yearConstructedRange',
              'livingSpaceRange'], axis=1)             
df = df[df.baseRent.between(100,10000, inclusive=True)] #drop extreme rent values
df = df[df.livingSpace.between(10, 500, inclusive=True)] #drop extreme and wrongly coded values
df = df[df.floor.between(-1, 50, inclusive=True)] #drop extreme and wrongly coded values
df = df[df.yearConstructed.between(1900, 2020, inclusive=True)] #drop extreme and wrongly coded values
df = df[df.noRooms.between(0,15, inclusive=True)] #drop extreme and probably wrongly coded value
df = df[np.isfinite(df['totalRent'])] #drop observations where totalRent isn't available
df = df[df.totalRent.between(100,10000, inclusive=True)] #drop extreme totalRent value

#df.to_excel("imm_data.xlsx", sheet_name='data')
datarows_current=df.shape[0]
print("We have {} appartments/rows".format(datarows_current))
print("Procentage of lost: {} ".format((1-datarows_current/datarows_original)*100))

# Setting a base map
gdf = geopandas.read_file("../input/germany-geo-json/gr.json")
gdf = gdf.drop(['ID_0','VARNAME_1','ISO','ISO','NAME_0','NL_NAME_1','TYPE_1','ENGTYPE_1'], axis=1) 
gdf.rename(columns={'NAME_1': 'State'}, inplace=True)
gdf
# calculating median price per state
df.rename(columns={'regio1': 'State'}, inplace=True)
medianPrice  = pd.DataFrame(df.groupby(['State'])['totalRent'].mean().astype(int))
medianPrice = medianPrice.reset_index()
medianPrice.columns = ['State', 'Median_price']
medianPrice
colormap = linear.OrRd_04.scale(
    medianPrice.Median_price.min(),
    medianPrice.Median_price.max())
colormap
medianPrice_dict = medianPrice.set_index('State')['Median_price']
medianPrice_dict
color_dict = {key: colormap(medianPrice_dict[key]) for key in medianPrice_dict.keys()}
color_dict
base_m = folium.Map([51.3, 9.5],tiles='cartodbpositron', zoom_start=6, control_scale=True, attr='Mapbox attribution',smooth_factor=None)

folium.GeoJson(
    gdf,
    name='Finnish cities',
    show=True,
    style_function=lambda feature: {
        'fillColor': color_dict[feature['properties']['State']], 
        'color': '#964B00',
        'weight': 1,
        'dashArray': '1, 1',
        'fillOpacity':0.8
    },
    highlight_function=lambda x: {'weight':3,
        'color': '#964B00',
        'fillOpacity': 1
    },
    
    tooltip=folium.features.GeoJsonTooltip(
        fields = ['State'],
        aliases=['State:'],
    )    
).add_to(base_m)

colormap.caption = 'Mean appartment rent price color scale'
colormap.add_to(base_m)


# add markers with basic information
fg = folium.FeatureGroup(name='State Info')
for index,row in df2.iterrows():
    fg.add_child(folium.Marker(location=[row[2], row[3]], tooltip=row[1] )) #icon=folium.Icon(color='darkblue', icon='point')
base_m.add_child(fg)

base_m.add_child(folium.LatLngPopup())
base_m.save('base_map.html')

base_m