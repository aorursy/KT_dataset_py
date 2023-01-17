# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as plt 

import re

import folium



%matplotlib inline 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#importing data 



df = pd.read_csv('/kaggle/input/oil-brazilian-beaches/2019-11-21-LOCALIDADES-AFETADAS.csv')

df.head()
#Understanding our data types

df.info()
#Renaming column names



df = df.rename(columns = {

    

    'localidade':'nome_praia',

    'Data_Avist':'dt_avistamento',

    'Data_Revis':'dt_revisao',

    'Status':'status_mancha',

    'Latitude':'lat',

    'Longitude':'lon'

    

})

df.head()
#Converting "dt_avistamento" to datetime and separating day, year and month



df['dt_avistamento'] = pd.to_datetime(df['dt_avistamento'])

df['mes'], df['dia'],df['ano'] = df['dt_avistamento'].dt.month, df['dt_avistamento'].dt.day, df['dt_avistamento'].dt.year
#Grouping "data_avistamento" and inserting it into a line chart to spot the growing oil sightings

dftime = df.groupby(['dt_avistamento']).size()

dftime.head(10)

dftime.plot(kind = 'line',figsize=(15,5))
#Viewing the months when the most spots appeared



dftime = df.groupby(['mes']).size()

dftime.plot(kind = 'bar',rot=True,figsize=(15,5))
#Viewing which states were most affected



dfestado = df.groupby(['sigla_uf']).size()

dfestado.plot(figsize=(15,5),kind = 'bar',color = 'coral',rot = True)
#Showing status of stains found on beaches

dfstatus = df.groupby(['status_mancha']).size()

dfstatus.plot(kind = 'pie',figsize=(20,8))
#How they had beaches that were hit more than once, and interesting to know the recidivism that these spots appeared



dfrec =  df.groupby(['nome_praia']).size().sort_values(ascending=False)

dfrec = dfrec.to_frame()

dfrec.reset_index(level=0, inplace=True)

dfrec = dfrec.rename(columns = {0:'recorrencia'})

dfrec = dfrec.query('recorrencia > 1')

dfrec.plot(figsize=(26,26),kind='barh',x = 'nome_praia', y = 'recorrencia',rot=True)

#This chart and a summarized version of the previous one, showing only which recidivisms appear more often



dfrec1 = dfrec.groupby(['recorrencia']).size().sort_values(ascending=False)

dfrec1.plot(kind = 'bar',rot=True,figsize=(15,6))
#Now let's check the location on a map of the affected places, for that I'm going to use folium

#and for that I'm going to have to start converting the latitude latitude and longitude degrees for decimals



def dms2dd(s):

    

    degrees, minutes, seconds, direction = re.split('[°\'"]+ ', s)

    dd = float(degrees) + float(minutes)/60 + float(seconds)/(60*60)

    

    if direction == 'S' or direction == 'W':

          dd *= -1

    return dd

#Now we apply our function in the date frame

df['lat'] = df['lat'].apply(dms2dd)

df['lon'] = df['lon'].apply(dms2dd)
#Setting an initial location for map view



locations = df[['lat', 'lon']]

locationlist = locations.values.tolist()

len(locationlist)

locationlist[7]
#Now displaying our map of places that have been hit by stains

#and also using color stain status marking



colors = {

        'Oleada - Vestigios / Esparsos': 'orange',

        'Oleada - Manchas': 'blue',

        'Oleo Nao Observado': 'green'

         }



brasil = folium.Map(

    location=[-17.974347 ,-39.472278],    

    zoom_start=5

)

for _, df in df.iterrows():

    if df['status_mancha'] in colors.keys():

     folium.Marker(

         location=[df['lat'], df['lon']],

         popup=df['nome_praia'],

         icon=folium.Icon(color=colors[df['status_mancha']])

     ).add_to(brasil)



brasil