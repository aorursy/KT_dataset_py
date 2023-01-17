import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import pandas as pd

import numpy as np

import folium

import requests

import json

import os
path_data = os.path.join('..', 'input', 'recorded-crime-data-at-police-force-area-level', 'rec-crime-pfa.csv')

path_geo = os.path.join('..', 'input', 'pfasgeo2017', 'pfas-geo-2017.json')



df = pd.read_csv(path_data)

df.head()
print(df.shape)

print(df.dtypes)
df['12 months ending'] = pd.to_datetime(df['12 months ending'], format='%d/%m/%Y')



df.head()
df['year'] = pd.DatetimeIndex(df['12 months ending']).year

df.drop(['12 months ending'], inplace=True, axis=1)

df.drop(['Region'], inplace=True, axis=1)



df.head()
df.rename(inplace=True, columns={

    'PFA': 'pfa',

    'Offence': 'offence',

    'Rolling year total number of offences': 'total'

})



df.head()
df.describe(include='all')
def quick_look(col_name):

    colunique = np.sort(df[col_name].unique())

    colnull = df[col_name].isnull().values.sum()

    

    print(colunique)

    print('Count unique:', len(colunique))

    print('Count null:', colnull)
quick_look('pfa')
rows_before = df.shape[0]



df = df[df['pfa'] != 'Action Fraud']

df = df[df['pfa'] != 'British Transport Police']

df = df[df['pfa'] != 'CIFAS']

df = df[df['pfa'] != 'UK Finance']



rows_after = df.shape[0]



quick_look('pfa')

print('Rows from:', rows_before, 'to:', rows_after, 'dropped:', (rows_before - rows_after))
quick_look('offence')
df.loc[df['offence'] == 'Domestic burglary', 'offence'] = 'Burglary'

df.loc[df['offence'] == 'Non-domestic burglary', 'offence'] = 'Burglary'

df.loc[df['offence'] == 'Non-residential burglary', 'offence'] = 'Burglary'

df.loc[df['offence'] == 'Residential burglary', 'offence'] = 'Burglary'



df.loc[df['offence'] == 'Bicycle theft', 'offence'] = 'Theft'

df.loc[df['offence'] == 'Shoplifting', 'offence'] = 'Theft'

df.loc[df['offence'] == 'Theft from the person', 'offence'] = 'Theft'

df.loc[df['offence'] == 'All other theft offences', 'offence'] = 'Theft'



df.loc[df['offence'] == 'Violence with injury', 'offence'] = 'Violence'

df.loc[df['offence'] == 'Violence without injury', 'offence'] = 'Violence'



df = df.groupby(['pfa', 'offence', 'year']).sum().reset_index()



quick_look('offence')
quick_look('total')
rows_before = df.shape[0]

df = df[df['total'] >= 0]

rows_after = df.shape[0]



quick_look('total')

print('Rows from:', rows_before, 'to:', rows_after, 'dropped:', (rows_before - rows_after))
df['total'].plot(kind='hist', color='red')

plt.show()
quick_look('year')
rows_before = df.shape[0]

df = df[df['year'] >= 2007]

rows_after = df.shape[0]



quick_look('year')

print('Rows from:', rows_before, 'to:', rows_after, 'dropped:', (rows_before - rows_after))
pfageo = os.path.abspath(path_geo)

print(pfageo)
def map_britain(data):

    britain = folium.Map(location=[52.355518, -1.174320],

                         zoom_start=6,

                         tiles='cartodbpositron')

    folium.Choropleth(

        geo_data=pfageo,

        name='choropleth',

        data=data,

        columns=['pfa', 'total'],

        key_on='feature.properties.pfa17nm',

        fill_color='YlGn',

        fill_opacity=0.7,

        line_opacity=0.2,

        legend_name='# Crimes'

    ).add_to(britain)



    folium.LayerControl().add_to(britain)

    return britain
label_weapon = 'Possession of weapons offences'

df_weapon = df.loc[df['offence'] == label_weapon]
map_britain(df_weapon.loc[df_weapon['year'] == 2018])
labels_weapon_high = ['Metropolitan Police', 'Greater Manchester', 'West Midlands', 'West Yorkshire']

df_weapon_high = df_weapon.loc[df_weapon['pfa'].isin(labels_weapon_high)]



sns.lineplot(data=df_weapon_high, x='year', y='total', hue='pfa')