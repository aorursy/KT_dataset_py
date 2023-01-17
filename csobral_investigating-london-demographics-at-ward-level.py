import pandas as pd

import geopandas as gpd

import matplotlib as mpl

import matplotlib.pyplot as plt



mpl.rcParams['axes.facecolor'] = 'white'

mpl.rcParams['figure.facecolor'] = 'white'
map_df = gpd.read_file('../input/london-borough-and-ward-boundaries-up-to-2014/London_Wards/Wards/London_Ward_CityMerged.shp')

print(map_df.columns)

print(f'Shapefile has {len(map_df)} entries')
map_df.head()
map_df.tail()
map_df = map_df[['NAME', 'BOROUGH', 'geometry']]

map_df = map_df.rename(columns={'NAME':'Ward', 'BOROUGH':'Borough'})

map_df
_, ax = plt.subplots(1, figsize=(10,10))

ax.axis('off')

map_df.plot(ax=ax)
wards_df = pd.read_csv('../input/london-ward-profiles-and-atlas/ward-profiles-excel-version.csv', encoding='unicode_escape')

print(wards_df.columns)
print(f'Ward profile dataset has {wards_df.columns.size} features and {len(wards_df)} entries')
df = wards_df[['Ward name', 'Population - 2015', 'Median Age - 2013']]

df = df.rename(columns={'Ward name':'Ward', 'Population - 2015':'Population', 'Median Age - 2013':'Median Age'})

df.head()
df.tail(10)
df = df[:-35]

print(f'Profile dataset now has {len(df)} entries')
df['Ward'].replace(to_replace=r'- .*', value='', regex=True)
df['Borough'] = df['Ward'].replace(to_replace=r'- .*', value='', regex=True)

df['Ward'].replace(to_replace=r'.*- ', value='', regex=True, inplace=True)

df.head()
map_df.head()
map_df.sort_values(by=['Borough','Ward'], inplace=True)

map_df.reset_index(drop=True, inplace=True)

df.sort_values(by=['Borough','Ward'], inplace=True)

df.reset_index(drop=True, inplace=True)



print('profile dset:\n', df.loc[:5, ['Ward', 'Borough']], '\n')

print('map dset:\n', map_df.loc[:5, ['Ward', 'Borough']])
#tot_df = map_df.join(df.drop('Borough', axis=1).set_index('Ward'), on='Ward')

tot_df = map_df.join(df.drop(['Ward','Borough'], axis=1))

tot_df.head()
from mpl_toolkits.axes_grid1 import make_axes_locatable



fig, ax = plt.subplots(1, figsize=(10,8))

plt.tight_layout()



divider = make_axes_locatable(ax)

cax = divider.append_axes("bottom", size="5%", pad=0.5)



ax.axis('off')

ax.set_title('Population of London by Electoral Ward')

tot_df.plot(column='Population', ax=ax, cax=cax, cmap='plasma', legend=True,

            legend_kwds={'orientation': 'horizontal', 'label': 'Population / Ward'})