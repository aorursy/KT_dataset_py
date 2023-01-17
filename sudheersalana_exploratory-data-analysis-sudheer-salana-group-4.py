# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import geopandas as gpd

import matplotlib.pyplot as plt

import xlrd

from shapely.geometry import Point

import contextily as ctx

import seaborn as sns

import geoplot.crs as gcrs

import geoplot as gplt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/cee-498-project-4-no2-prediction/train.csv')

print(df.shape)

print(df.info())
df.describe()
# First lets rearrange the data using the melt function



df_new1 = pd.melt(df, id_vars = ['Monitor_ID','State','Latitude', 'Longitude', 'Observed_NO2_ppb', 'WRF+DOMINO', 'Distance_to_coast_km','Elevation_truncated_km'], value_vars = df.loc[0:254, 'Impervious_100':'Impervious_10000'] , var_name = 'radius_m', value_name = 'impervious percentage')

df_new2 = pd.melt(df, id_vars = ['Monitor_ID','State','Latitude', 'Longitude', 'Observed_NO2_ppb', 'WRF+DOMINO', 'Distance_to_coast_km','Elevation_truncated_km'], value_vars = df.loc[0:254, 'Population_100':'Population_10000'], var_name = 'population', value_name = 'pop_number')

df_new3 = pd.melt(df, id_vars = ['Monitor_ID','State','Latitude', 'Longitude', 'Observed_NO2_ppb', 'WRF+DOMINO', 'Distance_to_coast_km','Elevation_truncated_km'], value_vars = df.loc[0:254, 'Major_100':'Major_10000'], var_name = 'major roads', value_name = 'maj_road_km')

df_new4 = pd.melt(df, id_vars = ['Monitor_ID','State','Latitude', 'Longitude', 'Observed_NO2_ppb', 'WRF+DOMINO', 'Distance_to_coast_km','Elevation_truncated_km'], value_vars = df.loc[0:254, 'Resident_100':'Resident_14000'], var_name = 'resident roads', value_name = 'res_road_km')

df_new5 = pd.melt(df, id_vars = ['Monitor_ID','State','Latitude', 'Longitude', 'Observed_NO2_ppb', 'WRF+DOMINO', 'Distance_to_coast_km','Elevation_truncated_km'], value_vars = df.loc[0:254, 'total_100':'total_14000'], var_name = 'total road', value_name = 'tot_road_km')



df_new1 = df_new1.set_index(['Monitor_ID','State','Latitude', 'Longitude', 'Observed_NO2_ppb', 'WRF+DOMINO', 'Distance_to_coast_km','Elevation_truncated_km', df_new1.groupby(['Monitor_ID','State','Latitude', 'Longitude', 'Observed_NO2_ppb', 'WRF+DOMINO', 'Distance_to_coast_km','Elevation_truncated_km']).cumcount()])

df_new2 = df_new2.set_index(['Monitor_ID','State','Latitude', 'Longitude', 'Observed_NO2_ppb', 'WRF+DOMINO', 'Distance_to_coast_km','Elevation_truncated_km', df_new2.groupby(['Monitor_ID','State','Latitude', 'Longitude', 'Observed_NO2_ppb', 'WRF+DOMINO', 'Distance_to_coast_km','Elevation_truncated_km']).cumcount()])

df_new3 = df_new3.set_index(['Monitor_ID','State','Latitude', 'Longitude', 'Observed_NO2_ppb', 'WRF+DOMINO', 'Distance_to_coast_km','Elevation_truncated_km', df_new3.groupby(['Monitor_ID','State','Latitude', 'Longitude', 'Observed_NO2_ppb', 'WRF+DOMINO', 'Distance_to_coast_km','Elevation_truncated_km']).cumcount()])

df_new4 = df_new4.set_index(['Monitor_ID','State','Latitude', 'Longitude', 'Observed_NO2_ppb', 'WRF+DOMINO', 'Distance_to_coast_km','Elevation_truncated_km', df_new4.groupby(['Monitor_ID','State','Latitude', 'Longitude', 'Observed_NO2_ppb', 'WRF+DOMINO', 'Distance_to_coast_km','Elevation_truncated_km']).cumcount()])

df_new5 = df_new5.set_index(['Monitor_ID','State','Latitude', 'Longitude', 'Observed_NO2_ppb', 'WRF+DOMINO', 'Distance_to_coast_km','Elevation_truncated_km', df_new5.groupby(['Monitor_ID','State','Latitude', 'Longitude', 'Observed_NO2_ppb', 'WRF+DOMINO', 'Distance_to_coast_km','Elevation_truncated_km']).cumcount()])



df3 = pd.concat([df_new1, df_new2, df_new3, df_new4, df_new5],axis=1)

df3 = df3.dropna()



df3['radius_m'] = df3['radius_m'].str.extract('(\d+)').astype(int)

df3['population'] = df3['population'].str.extract('(\d+)').astype(int)

df3['resident roads'] = df3['resident roads'].str.extract('(\d+)').astype(int)

df3['total road'] = df3['total road'].str.extract('(\d+)').astype(int)

df3['major roads'] = df3['major roads'].str.extract('(\d+)').astype(int)

df3

df3 = df3.drop(['population','resident roads','total road','major roads'], axis=1)

df3 = df3.reset_index()



df3 = df3.dropna(subset=['Latitude', 'Longitude'])

points = df3.apply(lambda row: Point(row.Longitude, row.Latitude), axis =1)

points

df3_new = gpd.GeoDataFrame(df3, geometry = points)

df3_new.set_crs(epsg=5070, inplace=True)
plt.figure(figsize = (12, 6))

ax = sns.boxplot(x='State', y='Observed_NO2_ppb', data=df3_new)

plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")

plt.xticks(rotation=45)
df3_new = df3_new.drop(['Latitude', 'Longitude', 'level_8'], axis=1)
df3_new2 = df3_new.drop(['Monitor_ID', 'State', 'radius_m'], axis=1)

print(df3_new2.corr()['Observed_NO2_ppb'])
newdata_2= df3_new[(df3_new != 0).all(1)]

df3_new3 = newdata_2.drop(['Monitor_ID', 'State','geometry'], axis=1)



for i in range(0, len(df3_new3.columns), 5):

    sns.pairplot(data=df3_new3,

                x_vars=df3_new3.columns[i:i+5],

                y_vars=['Observed_NO2_ppb'])
sns.set_style('ticks')

fig, ax = plt.subplots()



fig.set_size_inches(11.7, 8.27)

sns.boxplot(data=df3_new, x="radius_m", y="Observed_NO2_ppb", ax=ax)    

sns.despine()



df3_new2 = df3_new3.loc[df3_new3['radius_m'] == 800]

df3_new_3 = df3_new3.loc[df3_new3['radius_m'] == 500]

df3_new4 = df3_new3.loc[df3_new3['radius_m'] == 300]

df3_new5 = df3_new3.loc[df3_new3['radius_m'] == 1000]

df3_new6 = df3_new3.loc[df3_new3['radius_m'] == 1500]

df3_new7 = df3_new3.loc[df3_new3['radius_m'] == 2000]

df3_new8 = df3_new3.loc[df3_new3['radius_m'] == 3000]

df3_new9 = df3_new3.loc[df3_new3['radius_m'] == 4000]

df3_new10 = df3_new3.loc[df3_new3['radius_m'] == 5000]

df3_new11 = df3_new3.loc[df3_new3['radius_m'] == 6000]

df3_new12 = df3_new3.loc[df3_new3['radius_m'] == 7000]

df3_new13 = df3_new3.loc[df3_new3['radius_m'] == 8000]

df3_new14 = df3_new3.loc[df3_new3['radius_m'] == 9000]

df3_new15 = df3_new3.loc[df3_new3['radius_m'] == 10000]
print(df3_new2.corr()['Observed_NO2_ppb'])

print(df3_new_3.corr()['Observed_NO2_ppb'])

print(df3_new4.corr()['Observed_NO2_ppb'])

print(df3_new5.corr()['Observed_NO2_ppb'])

print(df3_new6.corr()['Observed_NO2_ppb'])

print(df3_new7.corr()['Observed_NO2_ppb'])

print(df3_new8.corr()['Observed_NO2_ppb'])

print(df3_new9.corr()['Observed_NO2_ppb'])

print(df3_new10.corr()['Observed_NO2_ppb'])

print(df3_new11.corr()['Observed_NO2_ppb'])

print(df3_new12.corr()['Observed_NO2_ppb'])

print(df3_new13.corr()['Observed_NO2_ppb'])

print(df3_new14.corr()['Observed_NO2_ppb'])

print(df3_new15.corr()['Observed_NO2_ppb'])
final_df_2 = pd.concat([df3_new11, df3_new12, df3_new13, df3_new14, df3_new15])

final_df_2 = final_df_2.drop(['radius_m'], axis=1)

print(final_df_2.shape)

for i in range(0, len(final_df_2.columns), 5):

    sns.pairplot(data=final_df_2,

                x_vars=final_df_2.columns[i:i+5],

                y_vars=['Observed_NO2_ppb'])



final_df_2.isnull().sum()
final_df_2.corr()['Observed_NO2_ppb']

fig = plt.figure(figsize =(15, 7)) 

boxplot = final_df_2.boxplot(column=[ 'Elevation_truncated_km', 'impervious percentage','maj_road_km','res_road_km','tot_road_km'])
corr = final_df_2.corr()

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);

plt.figure(figsize=(9, 8))

final_df_2.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
final_df_2[['Distance_to_coast_km', 'Elevation_truncated_km']].corr()  


fig = plt.figure(figsize =(15, 7)) 

boxplot = final_df_2.boxplot(column=[ 'pop_number'])





sns.pairplot(data=final_df_2,

                x_vars=['Distance_to_coast_km'],

                y_vars=['Elevation_truncated_km'])
print(final_df_2.corr()['Observed_NO2_ppb'])

print(df3_new15.corr()['Observed_NO2_ppb'])