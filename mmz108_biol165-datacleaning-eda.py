import numpy as np

import geopandas as gpd

import pandas as pd

import contextily as ctx

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

import mplleaflet

import os
def make_map(df, **kwargs):

    fig, ax = plt.subplots(figsize=(15,15))

    ax.axis('off')

    df.to_crs(epsg=3857).plot(ax=ax, **kwargs)

    ctx.add_basemap(ax)

    return ax
import os

print(os.listdir('../input/'))

os.chdir('../input/')
print(os.listdir('../input/biol165-p'))

print(os.listdir('../input/biol165-p2'))
litter = gpd.read_file("biol165-p/census_blocks_with_litter.shp")

income = gpd.read_file("biol165-p2/ACS_17_5YR_B19013.csv")
income = income.drop(0, axis=0)

income = income.drop(['GEO.display-label', 'GEO.id', 'HD02_VD01', 'geometry'], axis=1)

income.rename(columns={'GEO.id2': 'GEOID10', 'HD01_VD01':'MEDHI'}, inplace=True)

income['MEDHI'] = pd.to_numeric(income['MEDHI'])

income.head()
litter = litter.merge(income, on='GEOID10')
litter = litter[litter['MEDHI'].notnull()]
litter.head()
import pyproj

pyproj.Proj("+init=epsg:4326")
ax = make_map(litter, column='MEDHI', legend=True, cmap='Greens', alpha=0.6)

ax.set_title('Median Household Income by Location in Philadelphia')
from matplotlib.colors import LogNorm

plt.figure(figsize=(15,15))

ax = litter.plot(column='MEDHI', legend=True, cmap='Greens', alpha=0.6)

ax.set_title("Log Median Household Income by Census Block Group in Philadelphia")
trees = gpd.read_file('biol165-p/PPR_StreetTrees.shp')

trees = trees[['OBJECTID', 'geometry']]
census_blocks = gpd.read_file('biol165-p/Census_Block_Groups_2010.shp')

census_blocks = census_blocks[['GEOID10', 'geometry']]

census_blocks.head()
trees_w_tracts = gpd.sjoin(trees, census_blocks, how='inner', op='within')
trees_w_tracts = trees_w_tracts[['OBJECTID', 'geometry', 'GEOID10']]

trees_w_tracts.head()
litter['TREE_DENS'] = 0.3

for i_block, block in litter.iterrows():

    df = trees_w_tracts[trees_w_tracts['GEOID10'] == block['GEOID10']]

    #one square of lat-long is 3075.59 sq. miles

    density = len(df) / (3075.59*block['geometry'].area)

    litter.loc[i_block, 'TREE_DENS'] = density
litter.head()
make_map(litter, column='TREE_DENS', legend=True, cmap='Greens', alpha=0.6)
ipd = gpd.read_file('biol165-p2/DVRPC_IPD.csv', usecols=['GEOID10', 'IPD_SCORE'])
ipd['TRACTCE10'] = ipd['GEOID10'].str[5:]
litter = litter.merge(ipd, on='TRACTCE10')
litter
#  litter = gpd.read_file('litter_income_tree_ipd')
gsi = gpd.read_file('biol165-p/GSI_Public_Projects_Point.shp')
gsi = gsi[['OBJECTID', 'geometry']]

gsi = gsi.to_crs(epsg=3857)
litter['CENTROID'] = litter.to_crs(epsg=3857).centroid.apply(lambda p: np.array([p.x, p.y], dtype='float64'))
gsi['COORDS'] = gsi['geometry'].apply(lambda p: np.array([p.x, p.y], dtype='float64'))
data = np.array([s for s in gsi['COORDS']])

kde = stats.gaussian_kde(data.T, bw_method=.1)

block_group_centroids = np.array([s for s in litter['CENTROID']])

new_col = 1e9 * kde.evaluate(block_group_centroids.T)

litter['GSI_KDE'] = new_col
make_map(litter, column='GSI_KDE', alpha=0.7, cmap='Greens', legend=True)
litter = litter.drop(['CENTROID'], axis=1)
litter.to_file("litter_income_tree_ipd_gsi")

litter.to_file("litter_income_tree_ipd_gsi.csv", driver='CSV')