# Cargamos pandas, numpy y requests
import numpy as np  
import pandas as pd 

# libreria para obtener datos de la web
# http://docs.python-requests.org/en/master/
import requests

# Libreria para parsear paginas web 
# https://www.crummy.com/software/BeautifulSoup/bs4/doc/
from bs4 import BeautifulSoup

import datetime as dt
#df_historico_dolar = pd.read_csv('../input/historico-dolar.csv',index_col=0, parse_dates=True)
# Cargo los datos 
df_historico_dolar = pd.read_csv('../input/historico-dolar/historico-dolar.csv')
df_historico_dolar.head()
# remover la hora
df_historico_dolar['Fecha'] = df_historico_dolar['Fecha'].str.slice(0, 10)

# convertir a fecha 
df_historico_dolar['Fecha'] = pd.to_datetime(df_historico_dolar['Fecha'], dayfirst=True)
print(df_historico_dolar.dtypes)
# interpolar los datos
df_historico_dolar.interpolate()
df_historico_dolar.set_index('Fecha',inplace=True)
df_historico_dolar.head()
# Graficamos toda la serie
df_historico_dolar.plot()
#df_historico_dolar.Compra.plot()
# Las fechas multiples las agrupamos y buscamos el valor maximo por 
df_historico_dolar = df_historico_dolar.groupby(pd.Grouper(freq='D')).max().dropna()
df_historico_dolar.plot()
# Agroupar por año 2017
fecha_desde = dt.datetime(2017, 1, 1)
fecha_hasta = dt.datetime(2017, 12, 31)
df_historico_dolar_2017 = df_historico_dolar.loc[fecha_desde:fecha_hasta]
df_historico_dolar_2017.plot()
# Agroupar por año 2016
fecha_desde = dt.datetime(2016, 1, 1)
fecha_hasta = dt.datetime(2016, 12, 31)
df_historico_dolar_2016 = df_historico_dolar.loc[fecha_desde:fecha_hasta]
df_historico_dolar_2016.plot()
# Agroupar por año 2015
fecha_desde = dt.datetime(2015, 1, 1)
fecha_hasta = dt.datetime(2015, 12, 31)
df_historico_dolar_2015 = df_historico_dolar.loc[fecha_desde:fecha_hasta]
df_historico_dolar_2015.plot()
# Agroupar por año 2014
fecha_desde = dt.datetime(2014, 1, 1)
fecha_hasta = dt.datetime(2014, 12, 31)
df_historico_dolar_2014 = df_historico_dolar.loc[fecha_desde:fecha_hasta]
df_historico_dolar_2014.plot()
# Agroupar por año 2013
fecha_desde = dt.datetime(2013, 1, 1)
fecha_hasta = dt.datetime(2013, 12, 31)
df_historico_dolar_2013 = df_historico_dolar.loc[fecha_desde:fecha_hasta]
df_historico_dolar_2013.plot()
df_tipo_cambio = pd.read_csv('../input/tipo-cambio/datos-tipo-cambio-usd-futuro-dolar-frecuencia-diaria.csv')
df_tipo_cambio.head()
# convierto a fecha el indice_tiempo
df_tipo_cambio['indice_tiempo'] = pd.to_datetime(df_tipo_cambio['indice_tiempo'])
print(df_tipo_cambio.dtypes)
# pongo las fechas como indice
df_tipo_cambio.set_index('indice_tiempo',inplace=True)
df_tipo_cambio.head()
# Ahora voy a analizar el dolar a Mayorista (a3500)
df_tipo_cambio['tipo_cambio_a3500'].plot()

# Agroupar por año 2017
fecha_desde = dt.datetime(2017, 1, 1)
fecha_hasta = dt.datetime(2017, 12, 31)
df_tipo_cambio_2017 = df_tipo_cambio.loc[fecha_desde:fecha_hasta]
df_tipo_cambio_2017['tipo_cambio_a3500'].plot()