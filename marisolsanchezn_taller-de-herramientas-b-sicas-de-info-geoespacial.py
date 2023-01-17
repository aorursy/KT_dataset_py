#importamos las bases

import os
print(os.listdir("../input"))
#modulos necesarios
import numpy as np # funciones matemáticas
import pandas as pd # procesamiento de dataframes

#cargar base de viajes
Base1 = pd.read_json('../input/AllTrips.json') 
AllTrips = np.transpose(Base1) # transpone la base

#ver Trips en forma de dataframe(tabla)
AllTrips.head()    #Para saber el tamaño usar: print (AllTrips.shape)
#filtrar solo viajes de más de 0km
Trips = AllTrips[AllTrips["trip_distance"] > 0]
Trips.head()
#columnas requeridas para mapear las ubicaciones de inicio de los viajes

LatS = Trips['trip_start_latitude']
LonS = Trips['trip_start_longitude']
TimS = Trips['trip_start_timestamp']
#Cargamos un mapa disponible

import folium #modulo de mapas necesario
import folium.plugins #edición de mapas

Map1 = folium.Map(location=[19.42,-99.17],tiles="OpenStreetMap",max_zoom=18) #Otros mapas: Mapbox Bright,Mapbox Control Room, Stamen Toner, Stamen Terrain 
Map1
#Mapeamos los puntos de inicio

for k,l,m in zip(LatS,LonS,TimS): #ciclo for múltiple
    folium.CircleMarker(location=[k,l], 
                  popup = m,
                  color = "green",
                  fill_color = "green",
                  radius = 2 ).add_to(Map1)
    
Map1
## Pon tu código aquí


##Pon tu código aquí


#cargamos la base de sucursales
Branches = np.transpose(pd.read_json('../input/AllBranches.json'))
Branches.head()
#mapeamos las sucursales usando folium.marker y folium.icon
Map2 = folium.Map(location=[19.42,-99.17],tiles="OpenStreetMap",max_zoom=18) 

for k,l,m in zip(Branches['latitude'],Branches['longitude'],Branches['place']): #el popup no lee comillas
    folium.Marker(location=[k,l],popup = m, icon=folium.Icon(color='blue',icon='info-sign')).add_to(Map2)   
Map2
##Escribe aquí tu código
##Escribe tu código Aquí
##Escribe tu código aqui
#cargamos la base de clicks por sucursal
TableClicks = pd.read_json('../input/ClicksByBranch.json')
TableClicks.head()
#contamos el número de usuarios que han visto cada sucursal
UsersViews = pd.DataFrame(TableClicks.count(axis=0),columns=['UsersViews']) #transforma la lista en df

#Ordenamos
UsersViews.sort_values(by='UsersViews',ascending=False)
##Pon tu código aquí
##Escribe