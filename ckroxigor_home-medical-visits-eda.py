# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting
import seaborn as sns
sns.set(style="darkgrid")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
file_name = os.listdir("../input")[0]
df = pd.read_csv(os.path.join('..', 'input', file_name), parse_dates=[9])

print("The dataset has {} rows with {} features each".format(len(df), len(df.columns)))
df.head()

# Any results you write to the current directory are saved as output.
missing = []
for column in df.columns:
    missing.append({'column': column, 'missing': df[column].isnull().sum()})
missing = pd.DataFrame.from_records(missing)
missing[missing.missing > 0]
fig = sns.countplot(y="estado", data=df)
fig.set_title("Status count")
plt.show()
print("Average wait time: {} minutes".format(df.tiempo_demora.mean()))
fig = sns.countplot(y="tiempo_demora", data=df)
fig.set_title("Waiting time count")
fig.set_ylabel("Minutes")
plt.show()
fig = sns.boxplot(df.tiempo_demora)
fig.set_title("Waiting time count")
fig.set_xlabel("Minutes")
plt.show()
town_ids = df[['poblacion', 'id_poblacion']].drop_duplicates()
print("The same town may have different ids:")
(town_ids[town_ids.poblacion.apply(lambda name: name.strip().lower() == "barcelona")][:5])
towns_by_id = town_ids.groupby('id_poblacion').count().reset_index()
duplicate_ids = towns_by_id[towns_by_id.poblacion > 1].id_poblacion.tolist()
print("Towns sharing ids:")
town_ids[town_ids.id_poblacion.apply(lambda id: id in duplicate_ids)]
df[df.poblacion == 'CITY PROVES']
fig = sns.catplot(x="menor", y="edad_valor", data=df[~ df.edad_valor.isnull()])
fig.axes.flatten()[0].set_title("Age according to the 'under_age' (menor) value")
plt.show()
fig = sns.countplot(y="menor", data=df)
fig.set_title("Under age feature value count")
fig.set_ylabel("Value")
plt.show()
fig = sns.distplot(df[~ df.edad_valor.isnull()].edad_valor, kde=False)
fig.set_title("Patient age densitiy")
plt.show()
cps = df.cp
cps = cps[~cps.isnull()]
cps = cps.apply(lambda cp: str(int(cp)) if len(str(int(cp))) == 5 else '0' + str(int(cp)) )

plt.figure(figsize=(30,4))
fig = sns.countplot(cps, order = cps.value_counts().index)
fig.set_title("Occurrences per Postal Code")
fig.set_xlabel("Postal Code")
plt.xticks(rotation=90)
plt.show()
def parse_coord(value):
    if type(value) == str:
        return float(value.replace(',', '.'))
    return value

df.longitud_corregida = df.longitud_corregida.apply(parse_coord)
df.latitude_corregida = df.latitude_corregida.apply(parse_coord)
from mpl_toolkits.basemap import Basemap

# compute bounding box
margin = 10 # buffer to add to the range
lat_min = min(df.latitude_corregida) - margin
lat_max = max(df.latitude_corregida) + margin
lon_min = min(df.longitud_corregida) - margin
lon_max = max(df.longitud_corregida) + margin

# create map
m = Basemap(llcrnrlon=lon_min,
            llcrnrlat=lat_min,
            urcrnrlon=lon_max,
            urcrnrlat=lat_max)
m.drawcoastlines()
m.drawcountries()
m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color = 'white',lake_color='#46bcec')
# Add points
m.scatter(df.longitud_corregida, df.latitude_corregida, marker = 'o', color='r', zorder=5)
# Render
plt.show()
lat_limit = 20
lon_limit = 0

discarded = df[(df.latitude_corregida <= lat_limit) | (df.longitud_corregida <= lon_limit)]
print("{} entries outside Catalonia".format(len(discarded)))
discarded.head()

sliced = df[(df.latitude_corregida > lat_limit) & (df.longitud_corregida > lon_limit)]

# compute bounding box
margin = 2 # buffer to add to the range
lat_min = min(sliced.latitude_corregida) - margin
lat_max = max(sliced.latitude_corregida) + margin
lon_min = min(sliced.longitud_corregida) - margin
lon_max = max(sliced.longitud_corregida) + margin

# create map
m = Basemap(llcrnrlon=lon_min,
            llcrnrlat=lat_min,
            urcrnrlon=lon_max,
            urcrnrlat=lat_max)
m.drawcoastlines()
m.drawcountries()
m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color = 'white',lake_color='#46bcec')
# Add points
m.scatter(sliced.longitud_corregida, sliced.latitude_corregida, marker = 'o', color='red', zorder=5)
# Render
plt.show()
import folium
from folium import plugins
from folium.plugins import HeatMap

heatmap = folium.Map(location=[41.6888531,1.6248349], zoom_start = 8) 
HeatMap(sliced[['latitude_corregida', 'longitud_corregida']].values).add_to(heatmap)
heatmap
df[['patologia']].drop_duplicates().sample(15, random_state=42)
def format_month(d):
    return "{}-{}".format(d.year, d.month if d.month > 9 else "0" + str(d.month))

df['month'] = df.Fecha.apply(format_month)
plt.figure(figsize=(12,4))
fig = sns.countplot(df.month, order=sorted(df.month.unique()))
fig.set_title("Visits per month")
plt.xticks(rotation=70)
plt.show()
def format_hour(d):
    return "{}:00".format(d.hour if d.hour > 9 else "0" + str(d.hour))

hours = df.Fecha.apply(format_hour)
plt.figure(figsize=(12,4))
fig = sns.countplot(hours, order=sorted(hours.unique()))
fig.set_title("Visits per hours")
plt.xticks(rotation=30)
plt.show()
fig = sns.countplot(df.id_tipo)
fig.set_title("Count of visit types")
plt.show()
plt.figure(figsize=(16,4))
fig = sns.countplot(df.id_personal, order = df.id_personal.value_counts().index)
fig.set_title('Count by id_personal')
plt.xticks(rotation=70)
plt.show()
fig = sns.countplot(df.nasistencias)
fig.set_title("Count of number of visits")
plt.show()