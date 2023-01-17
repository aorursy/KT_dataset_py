# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/scrubbed.csv', low_memory=False)
df.sample()
df.info()
df['city']
#  La forma con mayor cantidad de registros en Los Angeles, CA es la forma de triangulo.
ciudades = df['city'].value_counts().to_frame()
ciudades.reset_index(inplace=True)
x = ciudades.loc[ciudades['index'] == 'los angeles', ]
print('los angeles, posicion {}'.format(x.index.values))
#  Existen 2184 registros de avistajes en Estados Unidos durante el año 2000.
df['datetime_x'] = df['datetime'].str.split().str[-2]
df['datetime_x'] = pd.to_datetime(df['datetime_x'])
x = df.loc[(df['datetime_x'].dt.year == 2000) & (df['country'] == 'us')]
len(x)
# El avistaje de mayor duración en segundos se registro en Canadá
df['duration (seconds)'] = pd.to_numeric(df['duration (seconds)'],errors='coerce')
df[['duration (seconds)','country']].sort_values('duration (seconds)',ascending=False)
#  La ciudad de Vancouver en Canadá tiene un registro en el Top20 de avistajes de mayor duración en Segundos. 
por_segundos = df[['duration (seconds)','city']].sort_values('duration (seconds)',ascending=False)
por_segundos.head(20)
# El tercer estado con mas avistajes de ovnis es Texas. 
ciudades = df['state'].value_counts().to_frame()
ciudades.reset_index(inplace=True)
ciudades
#x = ciudades.loc[ciudades['index'] == 'tx', ]
#x # Ojo porque los indices empiezan en 0!!!
#  Las Vegas, NV tiene un registro que aparece en el Top10 de avistajes con mayor duración en segundos para el mes de Marzo 2004. 
#df['datetime_x'] = df['datetime'].str.split().str[-2]
#df['datetime_x'] = pd.to_datetime(df['datetime_x'])
x = df.loc[(df['datetime_x'].dt.year == 2004) & (df['datetime_x'].dt.month == 3)]
x = x.sort_values('duration (seconds)', ascending=False)
x[['city']].head(10)
