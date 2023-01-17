# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

df_countries = pd.read_csv('/kaggle/input/countries-iso-codes/wikipedia-iso-country-codes.csv')
df_global = pd.read_csv('/kaggle/input/global-hospital-beds-capacity-for-covid19/hospital_beds_global_v1.csv')
df_global.dataframeName = 'hospital_beds_global_v1.csv'

df_global = df_global.merge(df_countries, how='left', left_on=['country'], right_on=['Alpha-2 code'])
df_global.rename(columns={'Alpha-3 code':'country code','English short name lower case': 'country name'}, inplace=True)

df_global = df_global[['country name', 'country code', 'beds', 'type', 'year', 'lat', 'lng', 'population']]
# df_global.head()

# Any results you write to the current directory are saved as output.
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
df_global_acute = df_global[df_global['type'] == 'ACUTE']

mapped = world.merge(df_global_acute[['country code', 'beds']], how='left', left_on='iso_a3', right_on='country code')
mapped = mapped.fillna(0)

to_be_mapped = 'beds'
vmin, vmax = 0,df_global_acute['beds'].max()
fig, ax = plt.subplots(1, figsize=(25,25))

mapped.plot(column=to_be_mapped, cmap='Blues', linewidth=0.8, ax=ax, edgecolors='0.8')
ax.set_title('Number of ACUTE beds per 1000 inhabitants in countries', fontdict={'fontsize':30})
ax.set_axis_off()

sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []

cbar = fig.colorbar(sm, orientation='horizontal')
df_global_icu = df_global[df_global['type'] == 'ICU']

mapped = world.merge(df_global_icu[['country code', 'beds']], how='left', left_on='iso_a3', right_on='country code')
mapped = mapped.fillna(0)

to_be_mapped = 'beds'
vmin, vmax = 0,df_global_icu['beds'].max()
fig, ax = plt.subplots(1, figsize=(25,25))

mapped.plot(column=to_be_mapped, cmap='Blues', linewidth=0.8, ax=ax, edgecolors='0.8')
ax.set_title('Number of ICU beds per 1000 inhabitants in countries', fontdict={'fontsize':30})
ax.set_axis_off()

sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []

cbar = fig.colorbar(sm, orientation='horizontal')
df_global_total = df_global[df_global['type'] == 'TOTAL']

mapped = world.merge(df_global_icu[['country code', 'beds']], how='left', left_on='iso_a3', right_on='country code')
mapped = mapped.fillna(0)

to_be_mapped = 'beds'
vmin, vmax = 0,df_global_total['beds'].max()
fig, ax = plt.subplots(1, figsize=(25,25))

mapped.plot(column=to_be_mapped, cmap='Blues', linewidth=0.8, ax=ax, edgecolors='0.8')
ax.set_title('Number of TOTAL beds per 1000 inhabitants in countries', fontdict={'fontsize':30})
ax.set_axis_off()

sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []

cbar = fig.colorbar(sm, orientation='horizontal')
df_cds = pd.read_csv('/kaggle/input/covidcdscountries/data_cds_countries.csv')

df_cds = df_cds[['country','iso3', 'deaths', 'cases']]
df_cds = df_cds.dropna()
df_cds = df_cds.groupby(["iso3"]).agg(
    {'cases':'last','deaths': "last", 'country': 'last'}).reset_index()
df_cds['mortality'] = df_cds.apply(lambda row:  row['deaths'] / row['cases'] * 100, axis=1)

df_mapped = df_cds.merge(df_global_icu[['country code', 'beds', 'lat', 'lng']], how='inner', left_on='iso3', right_on='country code')
def normalize(df, columns=None):
    result = df.copy()
    for feature_name in df.columns:
        if columns and feature_name not in columns:
            continue
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

map = folium.Map(location=[41.2925,12.5736], zoom_start=4)

# max_morality = df_mapped['mortality'].max()
# max_radius = 40
min_radius = 5

df_mapped_norm = normalize(df_mapped, ['mortality'])
    
for index, row in df_mapped_norm.iterrows():
    lat = row['lat']
    long = row['lng']

    mortality = row['mortality']
    beds = row['beds']
    
#     radius =  max(row['mortality'] * max_radius / max_morality, min_radius)
    radius = max(mortality * 30, min_radius)
#     print(mortality, beds, radius)

    if beds > 0.3:
        color = 'green'
    elif beds > 0.1:
        color = 'yellow'
    else:
        color = 'red'
    
    text = f'Beds pre 1000:{round(beds,3)}\nMortality:{round(row["mortality"],2)}%'
    folium.CircleMarker(location = [lat, long], color='black',fill = True, fill_color='grey', fill_opacity=0.5, radius=radius).add_to(map)
    folium.CircleMarker(location = [lat, long], color=color,fill = True, fill_color=color, fill_opacity=1, radius=5, popup=text).add_to(map)
map
df_mapped_most_affected = df_mapped[df_mapped.apply(lambda row: row['deaths'] > 1_000 or row['cases'] > 20_000, axis=1)]
df_mapped_most_affected.sort_values('mortality')
df = df_mapped_most_affected.nsmallest(10, ["mortality"])
df = df.set_index('country')
df.mortality.plot.bar(title="Top 10 countries with lowest mortality rate", align='center', legend="mortality")
df = df_mapped_most_affected.nlargest(10, ["beds"])
df = df.set_index('country')
df.beds.plot.bar(title="Top 10 countries by number of ICU beds per 1000", align='center', legend="ICU beds")
df = df_mapped_most_affected.set_index('country')
df = normalize(df[['beds', 'mortality']])
df = df.sort_values('beds')
df.plot.line(title="ICU beds number and mortality for each country (deaths > 1000)")
df_mapped_corr = df_mapped_most_affected[['beds','mortality']].corr()
plt.figure(figsize=(10,10))
plt.title('Correlation ICU beds/mortality (deaths > 1000)', fontsize=14)
sns.heatmap(df_mapped_corr, annot=True, fmt='.2f', square=True, cmap = 'Greens_r')