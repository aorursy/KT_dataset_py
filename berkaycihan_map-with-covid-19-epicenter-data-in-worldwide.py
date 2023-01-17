# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
!pip install folium
confirmed_csv='corona_confirmed.csv'

confirmed_gitpath= 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'


!curl -o $confirmed_csv $confirmed_gitpath


import pandas as pd

df=pd.read_csv(confirmed_csv)
df.head()
df=df.melt(id_vars=["Province/State","Country/Region","Lat","Long"],
           var_name="Date",
           value_name="Cases")
df.head()
df['Date']=df['Date'].str.replace(r'(/d+)/(\d+)/(\d+)', r'20\3-\1-\2')
df['Date']=pd.to_datetime(df['Date'])
df.head()
df['Province/State']=df['Province/State'].fillna(df['Country/Region'])
df.info()
df['Cases']=df['Cases'].astype(int)
df=df[df['Cases']>0].reset_index(drop=True)
df.head()
df_alarming_cities=df.sort_values(by='Cases', ascending=False).groupby('Country/Region').head(1).reset_index(drop=True)
df_alarming_cities=df_alarming_cities.head(n=10)
df_alarming_cities
import math

total_incidents=df['Cases'].sum()
def geojsons(df):
    features=[]
    for _, row in df.iterrows():
        feature ={
            'type':'Feature',
            'geometry':{
                'type':'Point',
                'coordinates':[row['Long'],row['Lat']]
        },
        'properties':{
            'time':pd.to_datetime(row['Date'],format='%Y-%m-%d').__str__(),
            'style':{'color':''},
            'icon':'circle',
            'iconstyle':{
                'fillColor':'red',
                'fillOpacity':0.8,
                'stroke':'true',
                'radius': math.log(row['Cases'])
                        }
                      }
       }
        features.append(feature)
    return features
start_geojson=geojsons(df)
start_geojson
import folium
from folium.plugins import TimestampedGeoJson

m=folium.Map(
    location=[50,30],
    zoom_start=2,
    tiles='Stamen Toner'
)
for _, row in df_alarming_cities.iterrows():
    
    folium.Marker(
    location=[row['Lat'],row['Long']],
    icon=folium.Icon(color='black',icon='ambulance', prefix='fa'),
    popup=row['Province/State']).add_to(m)
TimestampedGeoJson(
    start_geojson,
    period='P1D',
    duration='PT1M',
    transition_time=2000,
    auto_play=True,
).add_to(m)
m