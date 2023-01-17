from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import folium
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        file = (os.path.join(dirname, filename))

df = pd.read_excel(file, sheet_name=0, header=0, index_col='Unnamed: 0')

df = df.drop(['full_adress'], axis=1)
df
df.population_2010.groupby(df.area).sum()
df['Confirmed'] = 0

df['Recovered'] = 0

df['Deaths'] = 0



df.at[3, 'Confirmed'] = 1

df.at[3, 'Recovered'] = 1

df.at[57, 'Confirmed'] = 3

df.at[57, 'Recovered'] = 1

df.at[57, 'Deaths'] = 1

df.at[148, 'Confirmed'] = 1



#df['first_date_confirmed'] = ''

#df.at[3, 'first_date_confirmed'] = '15/03/2020'

#df.at[57, 'first_date_confirmed'] = '12/03/2020'

#df.at[148, 'first_date_confirmed'] = '20/03/2020'

#df['first_date_confirmed'] = pd.to_datetime(df['first_date_confirmed'],unit='ns')
confirmed = df.loc[df['Confirmed'] > 0]

confirmed.sort_values('Confirmed', ascending=False)
"""

Нужны данные: 

                A- начальное число контактов, 

                B- скорость распространения ∑/day?

                D- deaths rate? 

                С- скорость выздоровления

"""
#Get points

points = (df.lat.fillna(0),df.lot.fillna(0))

lat = points[0]

long = points[1]



# Map

map_tuva = folium.Map(location=[51.719082, 94.433983],width=750, height=500,max_zoom=7)

pop_title = df['population_2010']



for la,lo in zip(lat,long):

    folium.CircleMarker(

        location=[la,lo],

        radius=1,

        #popup='Population',

        color='green',

        fill=True,

        fill_color='green'

    ).add_to(map_tuva)

    

map_tuva