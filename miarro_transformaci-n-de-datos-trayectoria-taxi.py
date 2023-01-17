import pandas as pd

import numpy as np

import json

import geopandas as gpd

import csv

from shapely.geometry import LineString, Point
import os

ruta = ''

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        ruta = os.path.join(dirname, filename)

        

taxi = pd.read_csv(

    ruta,

    sep=",",

    low_memory=True,

    skiprows=lambda i: i % 100 != 0,  # Use only 1 of each n

)



taxi.head()
taxi.TRIP_ID.count()
taxi = taxi.drop_duplicates()

taxi.TRIP_ID.count()
taxi = taxi[taxi.MISSING_DATA == False]

taxi.TRIP_ID.count()
taxi = taxi[taxi.POLYLINE != "[]"]

taxi.TRIP_ID.count()
taxi.reset_index(drop=True, inplace=True)
taxi['TRAJECTORY'] = json.loads('[' + taxi.POLYLINE.str.cat(sep=',') + ']')

taxi = taxi[taxi.TRAJECTORY.str.len() > 1].copy()

taxi['LINES'] = gpd.GeoSeries(taxi.TRAJECTORY.apply(LineString))

taxi["TRIP_START"] = taxi['TRAJECTORY'].apply(lambda x: x[0])

taxi["TRIP_END"] = taxi["TRAJECTORY"].apply(lambda x: x[len(x)-1])

taxi.head()
taxi['COORDS_ini'] = json.loads('[' + taxi.TRIP_START.astype(str).str.cat(sep=',') + ']')

taxi['COORDS_fin'] = json.loads('[' + taxi.TRIP_END.astype(str).str.cat(sep=',') + ']')
taxi['LATITUDE_INI'] = taxi.COORDS_ini.apply(lambda x: x[1])

taxi['LONGITUDE_INI'] = taxi.COORDS_ini.apply(lambda x: x[0])

taxi['LATITUDE_FIN'] = taxi.COORDS_fin.apply(lambda x: x[1])

taxi['LONGITUDE_FIN'] = taxi.COORDS_fin.apply(lambda x: x[0])
taxi = taxi.drop(['TRIP_ID', 'CALL_TYPE', 'ORIGIN_CALL', 'ORIGIN_STAND', 'TAXI_ID', 'TIMESTAMP', 'DAY_TYPE', 'MISSING_DATA', 'POLYLINE', 'TRAJECTORY', 'LINES',  'TRIP_START', 'TRIP_END', 'COORDS_ini', 'COORDS_fin'], axis=1)
print(taxi)
def existe_nodo(dic, nodo):

    nodo_existente = -1

    for _id, value in dic.items():

        if (round(value['latitude'],3) == round(nodo['latitude'],3)) and (round(value['longitude'],3) == round(nodo['longitude'],3)):

            nodo_existente = _id

    return nodo_existente



nodos = {}

links = []

_id = 0

for indice_fila, fila in taxi.iterrows():

    print(indice_fila)

    #uniendo 2 nodos

    link_value = [] 

    nodo_actual = {'latitude' : fila['LATITUDE_INI'], 'longitude' : fila['LONGITUDE_INI']}

    nodo_existente = existe_nodo(nodos, nodo_actual)

    if nodo_existente != -1:

        link_value.append(_id)

    else:

        nodos[_id] = nodo_actual

        link_value.append(_id)

        _id = _id +1

    

    nodo_actual = {'latitude' : fila['LATITUDE_FIN'], 'longitude' : fila['LONGITUDE_FIN']}

    nodo_existente = existe_nodo(nodos, nodo_actual)

    if nodo_existente != -1:

        link_value.append(_id)

    else:

        nodos[_id] = nodo_actual

        link_value.append(_id)

        _id = _id +1

    

    if link_value[0] != link_value[1]:

        links.append(link_value)



print(len(nodos))

print(len(links))
with open('nodos.csv', 'w') as csvfile:

    writer = csv.writer(csvfile)

    row = ('id', 'latitude', 'longitude')

    writer.writerow(row)

    for _id, data in nodos.items():

        row = (_id, data['latitude'], data['longitude'])

        writer.writerow(row)

        

with open('aristas.csv', 'w') as csvfile:

    writer = csv.writer(csvfile)

    row = ('origen', 'destino', 'tipo')

    writer.writerow(row)

    for data in links:

        row = data[0], data[1], 'dirigida'

        writer.writerow(row)
print('FIN')
taxi.to_csv("trayectorias_taxi.csv", index=None)