import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import folium # Leaflet-style map visulization wrapper
import folium.plugins as plugins
from folium import LayerControl, Marker
# below is a dictionary pointing to custom icons used for the map
iconDictionary = {'Mining and Quarrying':'https://imgur.com/pUb5CPt.png', 
                  'Aluminum':'https://imgur.com/AZQGbKM.png',
                  'Metals (Except Aluminum and Iron and Steel)':'https://imgur.com/e6pBQI1.png',
                  'Oil sands':'https://imgur.com/wOkjXtf.png',
                  'Electricity':'https://imgur.com/ayZUN3e.png',
                  'Pulp and Paper':'https://imgur.com/AyPVCqk.png',
                  'Iron and Steel':'https://imgur.com/EjsL5Kv.png',                  
                  'Petroleum and Coal Product Refining and Mfg.':'https://imgur.com/UZoRfYP.png',
                  'Oil and gas extraction (excluding oil sands)':'https://imgur.com/vxr0qSA.png',
                  'Cement, Lime and Other Non-Metallic Minerals':'https://imgur.com/gwcG4BF.png',
                  'Wood Products':'https://imgur.com/3y0py82.png',
                  'Chemicals':'https://imgur.com/CBmc6Gb.png',
                  'Other Manufacturing':'https://imgur.com/u1huqtV.png',
                  'Other (Except Manufacturing)':'https://imgur.com/EUOwhsW.png',
                  'Oil and Gas Pipelines and Storage':'https://imgur.com/ibq3l3I.png',
                  'Water and Wastewater Systems':'https://imgur.com/nikvIVt.png',
                  'Transportation Equipment Mfg.':'https://imgur.com/6eR7n3c.png',
                  'Plastics and Rubber':'https://imgur.com/i67kU2Z.png',
                  'Waste Treatment and Disposal':'https://imgur.com/in2LgdM.png'}



#Read in the data as a pandas DataFrame rename a column and display the first three results to see structure
df = pd.read_csv("../input/NationalPollutantReleaseInventory.csv", encoding='latin1', index_col='Unnamed: 1')
df.rename(columns={'Unnamed: 0':'POLLUTANT_TYPE'}, inplace=True)
df[['POLLUTANT_TYPE','COMP_NAME','FACI_NAME','LATITUDE','LONGITUDE','NAICS4_NAME_EN','INTUITIVE_SECTOR_NAME_E','TOTAL_ALL_RELEASES']].head(3)
#A function to create the mapobj to display
def marker_heatmap_create(data):
    m = folium.Map([50, -100], zoom_start=4,tiles='stamentoner',world_copy_jump=True) #initializes the map object
    feature_heat = folium.FeatureGroup(name='Heatmap')                                #Feature groups used so layers can be turned off
    feature_group = plugins.MarkerCluster(name="Markers")                             #This plugin clusters the markers to help rendering on low zoom levels
    dataHeatmap = []
    #The below loop goes through every row of the dataFrame and creates the data for the heatmap while also creating markers and adding as a child of feature_group
    #dataHeatmap needs to be a list of lists with a lat, long and weight eg.[[lat1,long1,weight1],[lat2,long2,weight2]] if no weigth included does heatmap based on counts
    for _, series in data.iterrows():
        icon = folium.features.CustomIcon(iconDictionary[series['INTUITIVE_SECTOR_NAME_E']],icon_size=(16,16))
        coordinates = []
        coordinates.append([series['LATITUDE'],series['LONGITUDE'],series['TOTAL_ALL_RELEASES']])
        dataHeatmap.extend(coordinates)
        Marker(location=[series['LATITUDE'],series['LONGITUDE']],
                   popup=str(series['COMP_NAME'])+'<br>'+str(series['FACI_NAME'])+'<br>'+'Pollutant Type:'+str(series['POLLUTANT_TYPE'])+'<br>'+str(series['NAICS6_NAME_EN'])+'<br>'+str(series['TOTAL_ALL_RELEASES'])+' '+str(series['UNITS']),
                   icon=icon).add_to(feature_group)

    plugins.HeatMap(dataHeatmap).add_to(feature_heat)#creates the heatmap layer           
    feature_group.add_to(m)                          #adds marker cluster layer to map
    feature_heat.add_to(m)                           #adds heatmap layer to map
    LayerControl().add_to(m)                        #to control what layers are seen
    return m
marker_heatmap_create(df)  #displays the map, may take a few seconds to load
