#load the modules.
import numpy as np 
import pandas as pd 
import folium
from folium.plugins import HeatMap

from arcgis import *
df_with_addresses = pd.read_csv('../input/ds_analitycs_short.csv')
#We observe the struct the dataset
df_with_addresses.info()
#We checked the format of the addresses.
df_with_addresses['address'].head()
#We make a count of the fields for see how is distributed the records
df_with_addresses['address'].value_counts()[:15]
# We eliminate the records which contains the string 'Unknown'
df_with_addresses = df_with_addresses.loc[-(df_with_addresses['address'] \
                                            .str.contains('Unknown')),]
#convert value_counts() to list.
address = df_with_addresses['address'].value_counts().index.tolist()
# Print the qty of address grouped.
print("Qty of address grouped: {} ".format(len(address)))
#Init GIS
gis = GIS()
# We save the data in a dictionary. This is not strictly necessary. We could have saved the data in an array and then pass them 
#to Folium and it would work fine

#español
#Guardaremos los datos en un diccionario, esto no es estrictamente necesario podríamos haberlo guardado
#en un array y luego  pasabamos directamente al HeatMap y obtendriamos el mismo resultado. 
# Pero puede llegarnos a servir mantenerlo por algun motivo en particular.
dic_geocoordinates = {}
for i in address:
    georesult = geocode(i)[0]
    lng = georesult['attributes']['DisplayX'] 
    lat = georesult['attributes']['DisplayY']
    dic_geocoordinates[i] = dict(lng=lng, lat=lat)
georesult = geocode(address[0])[0]
# We control that the dictornary is complete
print("Qty of Address: {}".format(len(dic_geocoordinates)))
#Controlamos el formato.
dic_geocoordinates['São Paulo, Sao Paulo, Brazil']
#Now, we store the latitude and longitude in each record in the data set as appropriate.
df_with_addresses['lat'] = df_with_addresses['address'] \
                           .apply(lambda x: dic_geocoordinates[x]['lat'])
df_with_addresses['lng'] = df_with_addresses['address'] \
                           .apply(lambda x: dic_geocoordinates[x]['lng'])
#We control the  Dataset's format
df_with_addresses.head()
# We filter of the dataset the fields that we need
array_address = df_with_addresses.filter(items=['lat','lng'])
#We Control the quantity of the records
len(array_address)
#We start the Folium module and assign the values for X and Y in Brazil. And the zoom_start: 4.
mapHeat = folium.Map(location=[-13.7784,-55.9286], zoom_start=4)
#We pass the values and radius determines the intensity of each point that appears
mapHeat.add_child(folium.plugins.HeatMap(array_address.values, radius=10))
#Show the Map.
mapHeat
