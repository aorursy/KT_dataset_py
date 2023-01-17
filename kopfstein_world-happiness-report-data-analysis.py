import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
sns.set(color_codes=True)
sns.set_palette('colorblind')
%matplotlib inline
# path = '/tmp/data/world-happiness-report'   # path to local data
path = '../input/world-happiness'  # path to data in Kaggle notebook
dat2015 = pd.read_csv(os.path.join(path, '2015.csv'))
dat2016 = pd.read_csv(os.path.join(path, '2016.csv'))
dat2017 = pd.read_csv(os.path.join(path, '2017.csv'))
dat2015.head()
dat2017.head()
# 2017 data does not contain region, we create it based on the 2016 data:

# generating a new column using apply
# def get_region_by_country(country):
#    row = dat2016.query('Country == @country')
#    if row.shape[0] > 0:
#        return row.iloc[0].loc['Region']
#    else:
#        return np.nan
#
# dat2017['Region'] = dat2017.apply(lambda row: get_region_by_country(row['Country']), axis=1)

# creating region info using joins, much nice IMO
dat2017 = pd.merge(dat2017, dat2016.loc[:, ['Country', 'Region']], on='Country')
dat2017.rename(columns={'Happiness.Rank': 'Happiness Rank',
                       'Happiness.Score': 'Happiness Score',
                       'Economy..GDP.per.Capita.': 'Economy (GDP per Capita)',
                       'Health..Life.Expectancy.': 'Health (Life Expectancy)',
                       'Trust..Government.Corruption.': 'Trust (Government Corruption)', 'Dystopia.Residual': 'Dystopia Residual'}, inplace=True)
dat2017.head()
(any(dat2015.duplicated('Country').values)
    or any(dat2016.duplicated('Country').values)
    or any(dat2017.duplicated('Country').values))
dat2015.set_index('Country', inplace=True)
dat2016.set_index('Country', inplace=True)
dat2017.set_index('Country', inplace=True)
# shp_filename = shpreader.natural_earth(resolution='110m', category='cultural', name='admin_0_countries')
shp_filename = '../input/natural-earth/110m_cultural/ne_110m_admin_0_countries.shp'
shp_reader = shpreader.Reader(shp_filename)
country_name_map = {'Bosnia and Herz.': 'Bosnia and Herzegovina',
                    'Czechia': 'Czech Republic',
                    'Congo': 'Congo (Brazzaville)',
                    'Dem. Rep. Congo': 'Congo (Kinshasa)',
                    'Dominican Rep.': 'Dominican Republic',
                    'Greenland': 'Denmark',
                    'Palestine': 'Palestinian Territories',
                    'Somaliland': 'Somalia',
                   'United States of America': 'United States'}
plt.figure(figsize=(12,5))
ax = plt.axes(projection=ccrs.PlateCarree())
# ax.add_feature(cartopy.feature.OCEAN)
ax.set_extent([-150, 60, -25, 60])

map_colors = sns.color_palette('Blues_r', 8)

for country in shp_reader.records():
    if country.attributes['NAME'] in country_name_map:
        name = country_name_map[country.attributes['NAME']]
    else:
        name = country.attributes['NAME']
    if name in dat2017.index:
        ax.add_geometries(country.geometry, ccrs.PlateCarree(),
                          facecolor=map_colors[int(dat2017.loc[name, "Happiness Rank"] / (dat2017['Happiness Rank'].max() + 1) * len(map_colors))],
                          label=country.attributes['ADM0_A3'])
    else:
        ax.add_geometries(country.geometry, ccrs.PlateCarree(),
                          facecolor=(1, 0, 0),
                          label=country.attributes['ADM0_A3'])
        # print(name)
sns.kdeplot(dat2015['Happiness Score'], label='2015')
sns.kdeplot(dat2016['Happiness Score'], label='2016')
sns.kdeplot(dat2017['Happiness Score'], label='2017')
plt.xlabel('Happiness Score')
happiness_factors = ['Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)', 
           'Freedom', 'Generosity', 'Trust (Government Corruption)', 
           'Dystopia Residual']

def plot_columns_on_grid(data, columns, grid):
    for i, column in enumerate(columns):
        plt.subplot(grid[0], grid[1], i+1)
        sns.distplot(data[column])

plt.figure(figsize=(12,12))
plot_columns_on_grid(dat2017, happiness_factors, (3, 3))
dat = dat2017[happiness_factors].sum(axis=1)
residual = dat2017['Happiness Score'] - dat
residual.describe()
dat2017['Happiness Change'] = dat2017['Happiness Score'] - dat2015['Happiness Score']
dat2017['Happiness Change'].describe()
country_max_chg = dat2017['Happiness Change'].idxmax()
dat2017.loc[country_max_chg]
dat2017.loc[country_max_chg, happiness_factors] - dat2015.loc[country_max_chg, happiness_factors]
country_min_chg = dat2017['Happiness Change'].idxmin()
dat2017.loc[country_min_chg]
dat2017.loc[country_min_chg, happiness_factors] - dat2015.loc[country_min_chg, happiness_factors]
by_region = dat2017.groupby('Region')
by_region[['Happiness Score', 'Happiness Change'] + happiness_factors].mean().sort_values(by='Happiness Score', ascending=False)
sns.heatmap(by_region[happiness_factors[:-1]].mean().div(by_region['Happiness Score'].mean(), axis='index'))
dat2017_norm = dat2017
dat2017_norm[happiness_factors] = dat2017_norm[happiness_factors].div(dat2017['Happiness Score'].values, axis=0)
cluster_n = 3
k_means = KMeans(init='k-means++', n_clusters=cluster_n, n_init=10)
cluster_labels = k_means.fit_predict(dat2017_norm[happiness_factors[:-1]])
plt.figure(figsize=(12,12))
for i, factor in enumerate(happiness_factors):
    ax = plt.subplot(3, 3, i+1)
    for cluster in range(cluster_n):
        sns.kdeplot(dat2017_norm.loc[cluster_labels == cluster, factor], label=cluster)
        ax.set_title(factor)
for cluster in range(cluster_n):
    sns.kdeplot(dat2017.loc[cluster_labels == cluster, 'Happiness Score'], label=cluster)
dat2017['Cluster'] = cluster_labels
plt.figure(figsize=(12,5))
ax = plt.axes(projection=ccrs.PlateCarree())
# ax.add_feature(cartopy.feature.OCEAN)
ax.set_extent([-150, 60, -25, 60])

for country in shp_reader.records():
    if country.attributes['NAME'] in country_name_map:
        name = country_name_map[country.attributes['NAME']]
    else:
        name = country.attributes['NAME']
    if name in dat2017.index:
        ax.add_geometries(country.geometry, ccrs.PlateCarree(),
                          facecolor=sns.color_palette()[dat2017.loc[name, 'Cluster']],
                          label=country.attributes['ADM0_A3'])
    else:
        ax.add_geometries(country.geometry, ccrs.PlateCarree(),
                          facecolor=(1, 0, 0),
                          label=country.attributes['ADM0_A3'])  
