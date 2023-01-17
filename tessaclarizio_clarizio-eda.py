# Import helpful Python3 libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import descartes
import geopandas as gpd
from shapely.geometry import Point, Polygon


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_train = pd.read_csv('../input/cee-498-project-4-no2-prediction/train.csv')
df_train.head()
df_train.describe()
df_train.shape
df_train.columns
# Categorical features
print('Categorical: ', df_train.select_dtypes(include =['object']).columns)
# Numerical features
print('Numerical: ', df_train.select_dtypes(exclude=['object']).columns)
df_train.info()
# mean distribution
mu = df_train['Observed_NO2_ppb'].mean()
# std distribution
sigma = df_train['Observed_NO2_ppb'].std()
num_bins = 50

plt.figure(figsize=(8, 6))
n, bins, patches = plt.hist(df_train['Observed_NO2_ppb'], num_bins, density=True)

y = scipy.stats.norm.pdf(bins, mu, sigma)
plt.plot(bins, y, 'r--', linewidth=2)
plt.xlabel('Observed NO2 Concentration (ppb)')
plt.ylabel('Probability density')


plt.title(r'$\mathrm{Histogram\ of\ Observed\ NO2\ (ppb):}\ \mu=%.3f,\ \sigma=%.3f$' %(mu, sigma))
plt.grid(True)
#fig.tight_layout()
plt.show()
# First get a list of the names of states
states_list = df_train['State'].unique()
print(states_list)
print("There are " + str(len(states_list)) + " states represented in the dataset.") 
sns.set(rc={'figure.figsize':(16,8)}) # set the figure size of the boxplot to be larger than the default
sns.boxplot(x='State', y = 'Observed_NO2_ppb', data = df_train)
plt.title('Box plots of Observed NO2 (ppb) by State');
sns.boxplot(x='State', y = 'Observed_NO2_ppb', data = df_train)
sns.swarmplot(x='State', y = 'Observed_NO2_ppb', data = df_train, color = '.25')
plt.title('Box plots of Observed NO2 (ppb) by State with Observations');
df_train['State'].value_counts()
df_train['State'].value_counts().plot(kind='bar')
plt.xlabel('State')
plt.ylabel('Number of Monitors')
plt.title('Number of Monitors in each State');
# Create figure and axes
fig,ax = plt.subplots(1,1)

# Load a shapefile of the U.S.
states = gpd.read_file('../input/us-state-shp/us_state/tl_2017_us_state.shp')
states.plot(ax=ax, alpha=0.4, color='grey')

# First need to convert Longitude and Latitude to Points
df_points = df_train.apply(lambda row: Point(row.Longitude, row.Latitude), axis = 1)
# Now convert DataFrame to a GeoDataFrame
gpd_train = gpd.GeoDataFrame(df_train, geometry = df_points)
gpd_train.plot(column='Observed_NO2_ppb', ax=ax, alpha=0.5,legend=True, 
               markersize=12, legend_kwds={'label': 'Observed NO2 concentration (ppb)', 
                                          'orientation':'horizontal'})


# set latitude and longitude boundaries
plt.xlim(-130,-65)
plt.ylim(24,50)
plt.title('NO2 monitoring locations in the continental US')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
corr_coast = df_train.corr()['Observed_NO2_ppb']['Distance_to_coast_km']
corr_elevation = df_train.corr()['Observed_NO2_ppb']['Elevation_truncated_km']
print('The correlation between NO2 observations and distance to coast is ' + str(corr_coast) + '.')
print('The correlation between NO2 observations and elevation is ' + str(corr_elevation) + '.')
df_train.hist(figsize = (25,25), xlabelsize=4, ylabelsize = 4);
corr = df_train.corr()
ax = sns.heatmap(corr, vmin = -1, vmax =1, center = 0, 
                cmap=sns.diverging_palette(20,220,n=200), square=True)
ax.set_xticklabels(ax.get_xticklabels(), horizontalalignment='right');
df_roads = pd.melt(df_train, 
                   id_vars = ['Monitor_ID', 'State','Latitude','Longitude', 'Observed_NO2_ppb',
                              'WRF+DOMINO', 'Distance_to_coast_km', 'Elevation_truncated_km'], 
                   value_vars = ['total_100', 'total_200', 'total_300','total_400','total_500','total_600','total_700','total_800',
                               'total_1000','total_1200','total_1500','total_1800','total_2000','total_2500','total_3000','total_3500',
                               'total_4000','total_5000','total_6000','total_7000','total_8000', 'total_10000', 'total_10500', 'total_11000',
                               'total_11500', 'total_12000', 'total_12500', 'total_13000','total_13500', 'total_14000'], 
                   var_name = 'Total_roads_in_radius',
                   value_name = 'Total_roads_km')
df_roads
# First get dataframes with all of the total roads within a certain radius
df_roads_100 = df_roads[df_roads['Total_roads_in_radius']=='total_100']
df_roads_200 = df_roads[df_roads['Total_roads_in_radius']=='total_200']
df_roads_300 = df_roads[df_roads['Total_roads_in_radius']=='total_300']
df_roads_400 = df_roads[df_roads['Total_roads_in_radius']=='total_400']
df_roads_500 = df_roads[df_roads['Total_roads_in_radius']=='total_500']
df_roads_600 = df_roads[df_roads['Total_roads_in_radius']=='total_600']
df_roads_700 = df_roads[df_roads['Total_roads_in_radius']=='total_700']
df_roads_800 = df_roads[df_roads['Total_roads_in_radius']=='total_800']
df_roads_900 = df_roads[df_roads['Total_roads_in_radius']=='total_900']
df_roads_1000 = df_roads[df_roads['Total_roads_in_radius']=='total_1000']
df_roads_1200 = df_roads[df_roads['Total_roads_in_radius']=='total_1200']
df_roads_1500 = df_roads[df_roads['Total_roads_in_radius']=='total_1500']
df_roads_1800 = df_roads[df_roads['Total_roads_in_radius']=='total_1800']
df_roads_2000 = df_roads[df_roads['Total_roads_in_radius']=='total_2000']
df_roads_2500 = df_roads[df_roads['Total_roads_in_radius']=='total_2500']
df_roads_3000 = df_roads[df_roads['Total_roads_in_radius']=='total_3000']
df_roads_3500 = df_roads[df_roads['Total_roads_in_radius']=='total_3500']
df_roads_4000 = df_roads[df_roads['Total_roads_in_radius']=='total_4000']
df_roads_5000 = df_roads[df_roads['Total_roads_in_radius']=='total_5000']
df_roads_6000 = df_roads[df_roads['Total_roads_in_radius']=='total_6000']
df_roads_7000 = df_roads[df_roads['Total_roads_in_radius']=='total_7000']
df_roads_8000 = df_roads[df_roads['Total_roads_in_radius']=='total_8000']
df_roads_10000 = df_roads[df_roads['Total_roads_in_radius']=='total_10000']
df_roads_10500 = df_roads[df_roads['Total_roads_in_radius']=='total_10500']
df_roads_11000 = df_roads[df_roads['Total_roads_in_radius']=='total_11000']
df_roads_11500 = df_roads[df_roads['Total_roads_in_radius']=='total_11500']
df_roads_12000 = df_roads[df_roads['Total_roads_in_radius']=='total_12000']
df_roads_12500 = df_roads[df_roads['Total_roads_in_radius']=='total_12500']
df_roads_13000 = df_roads[df_roads['Total_roads_in_radius']=='total_13000']
df_roads_13500 = df_roads[df_roads['Total_roads_in_radius']=='total_13500']
df_roads_14000 = df_roads[df_roads['Total_roads_in_radius']=='total_14000']
# Next find correlation between roads and NO2
roads_100_corr = df_roads_100.corr()['Total_roads_km']['Observed_NO2_ppb']
roads_200_corr = df_roads_200.corr()['Total_roads_km']['Observed_NO2_ppb']
roads_300_corr = df_roads_300.corr()['Total_roads_km']['Observed_NO2_ppb']
roads_400_corr = df_roads_400.corr()['Total_roads_km']['Observed_NO2_ppb']
roads_500_corr = df_roads_500.corr()['Total_roads_km']['Observed_NO2_ppb']
roads_600_corr = df_roads_600.corr()['Total_roads_km']['Observed_NO2_ppb']
roads_700_corr = df_roads_700.corr()['Total_roads_km']['Observed_NO2_ppb']
roads_800_corr = df_roads_800.corr()['Total_roads_km']['Observed_NO2_ppb']
roads_900_corr = df_roads_900.corr()['Total_roads_km']['Observed_NO2_ppb']
roads_1000_corr = df_roads_1000.corr()['Total_roads_km']['Observed_NO2_ppb']
roads_1200_corr = df_roads_1200.corr()['Total_roads_km']['Observed_NO2_ppb']
roads_1500_corr = df_roads_1500.corr()['Total_roads_km']['Observed_NO2_ppb']
roads_1800_corr = df_roads_1800.corr()['Total_roads_km']['Observed_NO2_ppb']
roads_2000_corr = df_roads_2000.corr()['Total_roads_km']['Observed_NO2_ppb']
roads_2500_corr = df_roads_2500.corr()['Total_roads_km']['Observed_NO2_ppb']
roads_3000_corr = df_roads_3000.corr()['Total_roads_km']['Observed_NO2_ppb']
roads_3500_corr = df_roads_3500.corr()['Total_roads_km']['Observed_NO2_ppb']
roads_4000_corr = df_roads_4000.corr()['Total_roads_km']['Observed_NO2_ppb']
roads_5000_corr = df_roads_5000.corr()['Total_roads_km']['Observed_NO2_ppb']
roads_6000_corr = df_roads_6000.corr()['Total_roads_km']['Observed_NO2_ppb']
roads_7000_corr = df_roads_7000.corr()['Total_roads_km']['Observed_NO2_ppb']
roads_8000_corr = df_roads_8000.corr()['Total_roads_km']['Observed_NO2_ppb']
roads_10000_corr = df_roads_10000.corr()['Total_roads_km']['Observed_NO2_ppb']
roads_10500_corr = df_roads_10500.corr()['Total_roads_km']['Observed_NO2_ppb']
roads_11000_corr = df_roads_11000.corr()['Total_roads_km']['Observed_NO2_ppb']
roads_11500_corr = df_roads_11500.corr()['Total_roads_km']['Observed_NO2_ppb']
roads_12000_corr = df_roads_12000.corr()['Total_roads_km']['Observed_NO2_ppb']
roads_12500_corr = df_roads_12500.corr()['Total_roads_km']['Observed_NO2_ppb']
roads_13000_corr = df_roads_13000.corr()['Total_roads_km']['Observed_NO2_ppb']
roads_13500_corr = df_roads_13500.corr()['Total_roads_km']['Observed_NO2_ppb']
roads_14000_corr = df_roads_14000.corr()['Total_roads_km']['Observed_NO2_ppb']


roads_total_corr = pd.DataFrame({'correlation':[roads_100_corr,roads_200_corr, roads_300_corr, roads_400_corr,
                                                roads_500_corr, roads_600_corr, roads_700_corr, roads_800_corr,
                                                roads_900_corr, roads_1000_corr, roads_1200_corr, roads_1500_corr,
                                                roads_1800_corr, roads_2000_corr, roads_2500_corr, roads_3000_corr,
                                                roads_3500_corr, roads_4000_corr, roads_5000_corr, roads_6000_corr,
                                                roads_7000_corr, roads_8000_corr, roads_10000_corr, roads_10500_corr,
                                                roads_11000_corr, roads_11500_corr, roads_12000_corr, roads_12500_corr,
                                                roads_13000_corr, roads_13500_corr, roads_14000_corr]})
roads_total_corr.index = ['roads_100_corr', 'roads_200_corr', 'roads_300_corr', 'roads_400_corr',
                         'roads_500_corr', 'roads_600_corr', 'roads_700_corr', 'roads_800_corr',
                          'roads_900_corr', 'roads_1000_corr', 'roads_1200_corr', 'roads_1500_corr',
                          'roads_1800_corr', 'roads_2000_corr', 'roads_2500_corr', 'roads_3000_corr',
                          'roads_3500_corr', 'roads_4000_corr', 'roads_5000_corr', 'roads_6000_corr',
                          'roads_7000_corr', 'roads_8000_corr', 'roads_10000_corr',' roads_10500_corr',
                          'roads_11000_corr', 'roads_11500_corr', 'roads_12000_corr',' roads_12500_corr',
                          'roads_13000_corr', 'roads_13500_corr', 'roads_14000_corr']
roads_total_corr
roads_total_corr.idxmax()