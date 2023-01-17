import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

ubicacion= "/kaggle/input/spanish-high-speed-rail-system-ticket-pricing/high_speed_spanish_trains.csv"
datos= pd.read_csv(ubicacion)
datos.head()
datos.info()
#normalizar registros
datos['price_tree'] = datos['price_tree'].fillna(0.0)
datos['price'] = datos['price'].fillna(0.0)
datos['train_class'] = datos['train_class'].fillna('None')
datos['fare'] = datos['fare'].fillna('None')
datos['batch'] = datos['batch'].fillna('None')

#normalizar tipos de datos
datos['insert_date'] = pd.to_datetime(datos['insert_date']) 
datos['origin'].astype(str)
datos['destination'].astype(str)
datos['start_date'] = pd.to_datetime(datos['start_date']) 
datos['end_date'] = pd.to_datetime(datos['end_date']) 
datos['train_type'].astype(str)
datos['price'].astype(float)
datos['train_class'].astype(str)
datos['fare'].astype(str)
datos['price_tree'].astype(str)
datos['batch'].astype(str)
datos['id'].astype(str)
datos = datos.sample(frac=0.30, random_state=99)
datos
print("***Estadisticos para precio***")
print("Media: ",datos['price'].mean())
print("Desviacion Estandar: ",datos['price'].std())
print("Moda: ",datos['price'].mode())
datos['start_date_year'] = pd.DatetimeIndex(datos['start_date']).year
datos['start_date_month'] = pd.DatetimeIndex(datos['start_date']).month
datos['start_date_day'] = pd.DatetimeIndex(datos['start_date']).day
start_date_polarity = datos.groupby(["origin", 'start_date_year', 'start_date_month','start_date_day'], as_index=False).count()
start_date_polarity.head(10)
datos['train_class'].value_counts()
medias= datos.groupby([datos['start_date_month']])['destination'].value_counts()
medias
tarifa1= datos.groupby([datos['train_class']])['price'].mean()
tarifa1.index.names= ["CLASE"]
tarifa1
tarifa2= datos.groupby([datos['start_date_month'], datos['destination']])['price'].mean()
tarifa2.index.names= ["MES","DESTINO"]
tarifa2
tarifa3= datos.groupby([datos['destination']])['price'].min()
tarifa3.index.names= ["DESTINO"]
print("***MIN***")
tarifa3
tarifa4= datos.groupby([datos['destination']])['price'].max()
tarifa4.index.names= ["DESTINO"]
print("***MAX***")
tarifa4