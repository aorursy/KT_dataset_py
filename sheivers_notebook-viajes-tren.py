# Andres Camilo Castañeda Barrios - T00047921

# Giovanny Zdenco Jukopila Rueda - T00041851



# Importamos las librerias

import os

import io

import sys

import os

import random

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from plotly.offline import iplot
# Importamos el Dataset

data = pd.read_csv('../input/spanish-high-speed-rail-system-ticket-pricing/high_speed_spanish_trains.csv')
# Primero miramos la informacion base del dataset

# De aqui podemos obtener los tipos de datos de las columnas, la cantidad de columnas, el tamaño del objeto y la cantidad de registros

data.info()
# Tomamos una muestra de los datos para anilizar cuales son las columnas con las que vamos a trabajar ( 5 datos )

data.head(5)
# Normalizamos los datos

data['price_tree'] = data['price_tree'].fillna(0.0)

data['price'] = data['price'].fillna(0.0)

data['train_class'] = data['train_class'].fillna('None')

data['fare'] = data['fare'].fillna('None')

data['batch'] = data['batch'].fillna('None')



# Normalizamos los tipos de datos

data['insert_date'] = pd.to_datetime(data['insert_date']) 

data['origin'].astype(str)

data['destination'].astype(str)

data['start_date'] = pd.to_datetime(data['start_date']) 

data['end_date'] = pd.to_datetime(data['end_date']) 

data['train_type'].astype(str)

data['price'].astype(float)

data['train_class'].astype(str)

data['fare'].astype(str)

data['price_tree'].astype(str)

data['batch'].astype(str)

data['id'].astype(str)
# Obtenemos los valores de interes



# Tarifa mínima y máxima por destino.

print( "El precio maximo de un tiquete fue de", data.price.max(), "Euros" )

print( "El precio minimo de un tiquete fue de", data.price.min(), "Euros" )

print( "El precio promedio de los tiquetes fue de", np.round( data.price.mean(),2 ), "Euros" )





# Media, moda y mas de datos numericos

data[{'price': ['min', 'max', 'median', 'skew']}].describe()
# Datos de destinos

origenes = data.origin.unique()

destinos = data.destination.unique()



# Obtenemos las ciudades que cuentas con al menos una ruta

ciudades = [next(iter(filter(None, values)), '') for values in zip(origenes, destinos)]

print( "Estas son las ciudades que cuentas con vias de tren registradas para este conjunto de datos \n", ciudades )
# Trabajamos sobre una muestra mas reducida

dataShort = data.sample( frac=0.30, random_state=99 )
# Convertimos los datos de las columnas de fechas en datos datetime, para poder manejarlos mejor

for i in ['insert_date','start_date','end_date']:

    dataShort[i] = pd.to_datetime(dataShort[i])
# Cargamos las columnas de interes en columnas propias



dataShort['horaInicio'] = dataShort['start_date'].dt.hour

dataShort['diaInicio'] = dataShort['start_date'].dt.day

dataShort['mesInicio'] = dataShort['start_date'].dt.month

dataShort['anioInicio'] = dataShort['start_date'].dt.year



dataShort['horaFin'] = dataShort['end_date'].dt.hour

dataShort['diaFin'] = dataShort['end_date'].dt.day

dataShort['mesFin'] = dataShort['end_date'].dt.month

dataShort['anioFin'] = dataShort['end_date'].dt.year



dataShort['tiempoTotal'] = dataShort['end_date'] - dataShort['start_date']

dataShort['minutos'] = dataShort['tiempoTotal']/np.timedelta64(1,'m')



dataShort.head()

# Obtenemos informacion de meses

meses = dataShort['mesInicio'].value_counts()

plt.figure(figsize=(12,4))

sns.barplot(meses.index, meses.values, alpha=0.8)

plt.ylabel('Numero de viajes (Millones)', fontsize=10)

plt.xlabel('Mes', fontsize=12)

plt.show()
# Obtenemos informacion de horas

print( "En promedio, la duracion de los viajes es de ",round(dataShort['minutos'].mean(),2),"minutos" )

print( "Los viajes que mas demoran son de mas o menos",round(dataShort['minutos'].max(),2),"minutos" )

print( "Los viajes mas rapidos son de mas o menos ",round(dataShort['minutos'].min(),2),"minutos" )
# Analizamos los tipos de trenes



trenes = dataShort['train_type'].value_counts()

etiquetas = list(trenes.index)



plt.pie(trenes, labels=etiquetas,radius=3, autopct='%1.2f%%')

plt.legend(labels = etiquetas, loc="center right")



plt.show()