import numpy as np
import pandas as pd
from subprocess import check_output
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import os
os.listdir("./")
df = pd.read_csv('../input/worldcitiespop.csv', encoding='latin-1', dtype={
    'Country': str,
    'City': str,
    'AccentCity': str,
    'Region': str
})[['Country', 'City', 'Region', 'Population', 'Latitude', 'Longitude']]
df.info()
df.head()
print(len(df.Country.unique()), "countries")
print(len(df.City), "cities")
df.apply(lambda x: x.isnull().sum())
df_pop = df.dropna(how='any')
df_pop["Country"] = df_pop["Country"].astype(str).str.upper()
plt.figure(figsize=(12,5))
df_pop.groupby('Country').sum()['Population'].sort_values(ascending=False).head(20).plot.bar()
plt.xlabel('country')
plt.ylabel('population (million)')
plt.title('World population by country')
plt.show()
plt.figure(figsize=(30,12))
plt.scatter(df_pop['Longitude'], df_pop['Latitude'], c=np.log(df_pop['Population']), cmap='OrRd', s=.1)
plt.colorbar()
plt.title('Population distribution on Earth')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.savefig('worldpop.png')
plt.show()
plt.figure(figsize=(20,10))
df_hu = df_pop[df_pop.Country == 'HU']
plt.scatter(df_hu['Longitude'], df_hu['Latitude'], c=np.log(df_hu['Population']), s=df_hu['Population']/800)
plt.colorbar()
plt.title('Population distribution (logarithmic) in Hungary')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.show()
# left bottom 45.821990, 16.136645
# right up 48.487148, 23.052344
fig, ax = plt.subplots(figsize=(12,12))
m = Basemap(llcrnrlon=16.136645,llcrnrlat=45.821990,
            urcrnrlon=23.052344,urcrnrlat=48.487148,
              projection='cyl')
m.drawmapboundary(fill_color='lightblue')
m.drawcountries()
m.fillcontinents(color='#f2f2f2',lake_color='#46bcec')
plt.show()
