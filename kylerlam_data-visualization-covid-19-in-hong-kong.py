import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib.style

print(plt.style.available) # see what styles are available

mpl.style.use('Solarize_Light2')
df = pd.read_csv('../input/confirmed-cases-hk/enhanced_sur_covid_19_eng.csv')

df.head()
df.shape
df.columns
df.set_index('Case no.', inplace = True)

df
df.rename(columns = {'Case classification*': 'Case classification'}, inplace = True)
df['Name of hospital admitted'].value_counts(dropna = False)
# So, we could drop the entire column.

df.drop('Name of hospital admitted', axis = 1, inplace = True)
df['Case classification'].value_counts()
df.head()
df['Date of onset'].value_counts()
df_asym = df.copy()

df_asym = df_asym.replace(to_replace=r"^(.(?<!Asymptomatic))*?$", value = "Symptomatic", regex = True)

fig = plt.figure(figsize=(6,6), dpi=100)

df_asym['Date of onset'].value_counts().plot.pie(startangle = 90, autopct = '%1.1f%%', labels = None, explode = (0.1, 0))

plt.legend(labels = df_asym['Date of onset'].value_counts().index, bbox_to_anchor =(1, 0.5))

plt.show()
df_gender = df['Gender'].value_counts()

df_gender
# Gender Plot

fig = plt.figure(figsize=(6,6), dpi=100)

colors = ['#fc4f30','#008fd5']

plt.title('Gender')

df_gender.plot(kind = 'pie',labels = None ,colors = colors, startangle = 90, autopct='%1.2f%%',pctdistance=0.7, explode = (0.01, 0.01))

plt.legend(labels=df_gender.index, bbox_to_anchor =(1, 0.5)) 

plt.axis('equal')

plt.show()
# Age Plot

count, bin_edges = np.histogram(df['Age'], 20)

plt.style.use('seaborn-whitegrid')

plt.title('Age')

df['Age'].plot(kind = 'hist', figsize = (10,8), bins =bin_edges, edgecolor = 'black', linewidth = 1.2)

plt.xlabel("Age")

plt.show()
df_residence = df['HK/Non-HK resident'].value_counts()

df_residence
df.replace({'HK resident' : 'HK Resident', 'Non-HK resident':'Non-HK Resident'}, inplace = True)

df_residence = df['HK/Non-HK resident'].value_counts()

df_residence
fig = plt.figure(figsize=(6,6), dpi=100)

df_residence.plot(kind = 'pie', autopct = '%1.2f%%', pctdistance = 1.1, startangle = 90, labels = None, explode = (0.3, 0.3, 0))

plt.legend(labels=df_residence.index, loc='center left', shadow = True, bbox_to_anchor =(1, 0.5))

plt.axis('equal')



plt.show()
fig = plt.figure(figsize=(6,6), dpi=100)

df['Case classification'].value_counts().plot(kind = 'pie', autopct = '%1.1f%%', pctdistance = 1.15, startangle = 90, labels = None, explode = (0,0.05,0.1,0.15,0.2,0.25))

plt.legend(labels = df['Case classification'].value_counts().index, bbox_to_anchor =(1, 0.5))

plt.show()
date = df['Report date']

df_date = pd.DataFrame(date.value_counts())

df_date.reset_index(inplace = True)

df_date.columns = ['Report date', 'Cases']

df_date['Report date'] = pd.to_datetime(df_date['Report date'], format = '%d/%m/%Y')

df_date.sort_values(['Report date'], inplace = True)

df_date.set_index('Report date', inplace = True)

df_date['Cumulative Cases'] = df_date['Cases'].cumsum()
fig, ax1 = plt.subplots(figsize=(20,10))

ax1.set_ylabel('Daily Confirmed Cases', color = '#fc4f30')

ax1.set_xlabel('Date')

ax1 = df_date['Cases'].plot(color = '#fc4f30')



ax2 = ax1.twinx()

ax2.set_ylabel('Cumulative Confirmed Cases', color = '#008fd5')

ax2 = df_date['Cumulative Cases'].plot(kind = 'area', alpha = 0.3, color = '#008fd5')



ax2.grid(False)
#import another dataset

df_building = pd.read_csv('../input/covid19-building-list/building_list_eng.csv')
df_building.head()
df_building.shape
df_building.describe()
df_district = pd.DataFrame(df_building['District'].value_counts())

df_district.reset_index(inplace = True)

df_district.columns = ['District', 'Cases Count']

df_district.replace({'Central & Western' : 'Central and Western'}, inplace = True)

df_district
import folium

import geojson

hk_map = folium.Map(location = [22.34, 114.1], zoom_start = 11, tiles = 'cartodbpositron')

hk_map
hk_geo = geojson.load(open('../input/hk-geo/hkg_adm1.geojson.json'))
choropleth = folium.Choropleth(geo_data = hk_geo,

                  name = 'choropleth',

                  data = df_district, 

                  columns = ['District', 'Cases Count'],

                 key_on = 'feature.properties.name_1',

                 fill_color = 'PuRd',

                 legend_name = 'Cases Count',

                 highlight = True).add_to(hk_map)



toollip = folium.features.GeoJsonTooltip(fields = ['name_1'], aliases = ['District: '])

choropleth.geojson.add_child(toollip)



hk_map