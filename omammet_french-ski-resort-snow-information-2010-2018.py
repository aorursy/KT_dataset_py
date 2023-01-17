# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import datetime
from pandas.io.json import json_normalize
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from sklearn.cluster import AgglomerativeClustering
csv_file_postes = '../input/geographical-information-for-ski-resort-locations/postesNivo.csv'
df_postes = pd.read_csv(csv_file_postes, sep=',')

print("The dataset contains {0} weather station".format(df_postes.shape[0]))
print('-'*80)
print(df_postes.head())
print('-'*80)
print(df_postes.columns)
print('Altitude distribution')
print('-'*80)

fig1 = plt.figure(num=0,figsize=(8,8),dpi=80)
plt.title('Altitudes distribution')
plt.legend()
plt.xlabel('Altitude (m)')
plt.ylabel('Cohort')

plt.hist(df_postes['Altitude'], bins=25)

print(df_postes['Altitude'].describe())

# Plotting scatter plots of postes with altitude
fig2 = plt.figure(num=1,figsize=(8,8),dpi=80)
plt.title('Geographical distribution')
plt.legend()
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.scatter(df_postes['Longitude'], df_postes['Latitude'], c=df_postes['Altitude'], cmap='viridis')
plt.colorbar()
# Clustering of postes by geographical areas
n_clusters=7
cluster = AgglomerativeClustering(n_clusters=n_clusters)
cluster.fit_predict(df_postes[['Longitude','Latitude']])
print(cluster.labels_)

fig3 = plt.figure(num=2,figsize=(8,8),dpi=80)
plt.title('Geographical clustering')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.scatter(df_postes['Longitude'], df_postes['Latitude'], c=cluster.labels_)
plt.colorbar()
dict_clusters = {
        0 : 'Northern Alps',
        1 : 'Western Pyreneans',
        2 : 'Southern Alps',
        3 : 'Eastern Pyreneans',
        4 : 'Central Alps',
        5 : 'Corsica',
        6 : 'Central Massif'
        }
df_postes['cluster'] = cluster.labels_

# Finally plotting altiude distribution per cluster

alt_clust = []
for i in range(n_clusters):
    alt_clust.append(df_postes.loc[df_postes['cluster'] == i]['Altitude'])
    
fig4 = plt.figure(num=3, figsize=(8,8), dpi=80)
plt.boxplot(alt_clust)
plt.title('Altitude distribution per cluster')
plt.ylabel('Altitude (m)')
plt.xlabel('cluster')
#plt.xticks(ticks=list(range(1,8)), labels=list(dict_clusters.values()), rotation='vertical')
# To facilitate further clustersing, creating a dictionary of 'station number' : 'cluster number'
dict_numer_sta_cluster = {}
for index, row in df_postes.iterrows():
    dict_numer_sta_cluster[row['ID']] = row['cluster']
# Loading of the snowfall datafile
csv_file = '../input/french-ski-resort-snow-data-2010-2018/combinedfile.csv'
df = pd.read_csv(csv_file, sep=';')
# First renaming the df_postes ID name to be consistent with the main dataframe

# Removing the off row in the dataset and then print the many different columns

df = df[df['date'] != 'date']
df['numer_sta'] = df['numer_sta'].astype(int)
df_postes['numer_sta'] = df_postes['ID']
df = pd.merge(df, df_postes, on='numer_sta')
# Dataset description
df.columns
# Checking datatypes of the dataset
df.dtypes
# Reading the first few rows of the dataset
df.head()
# Converting dates to datetime
df['date'] = df['date'].astype(str)
df['date'] = pd.to_datetime(df['date'],format='%Y%m%d%H%M%S')

# Replacing 'mq' values by NaN when the measure is non existant
# Replacing 'ht_neige' values by NaN when the measure is non existant
# Total snow depth
df['ht_neige'] = df['ht_neige'].replace('mq','NaN')
df['ht_neige'] = df['ht_neige'].astype(float)
sample_station = 7875

# Fresh snow depth
df['ssfrai'] = df['ssfrai'].replace('mq','NaN')
df['ssfrai'] = df['ssfrai'].astype(float)

# Station altitude
df['haut_sta'] = df['haut_sta'].replace('mq','NaN')
df['haut_sta'] = df['haut_sta'].astype(float)
df_sample = df[df['numer_sta'] == sample_station]
# df_sample['ht_neige'].plot()
print('Description of snow depth series')
df_sample['ht_neige'].describe()
print('Description of new snow depth series')
df_sample['ssfrai'].describe()
df_sample.columns
fig4 = plt.figure(num=3,figsize=(8,8),dpi=80)
plt.title('Snow height at station {} in (m)'.format(sample_station))
plt.plot(df_sample['date'], df_sample['ht_neige'])
plt.plot(df_sample['date'], df_sample['ssfrai'])
# Dropping the time information
df_no_time = df
df_no_time['date'] = df_no_time['date'].dt.date
df_no_time.head()
# Average data across the dataset for each day
df_average = df_no_time.groupby(df_no_time['date']).mean()
df_average.describe()
df_average.head()
fig5 = plt.figure(num=4,figsize=(8,8),dpi=80)
plt.title('Average snow height (m) across the whole dataset')
plt.plot(df_average['ht_neige'])
dates_years = {}
mask_years = {}

for y in range(2011,2020):
    dates_years[y] = datetime.datetime.strptime(str(y)+'0101','%Y%m%d').date()
    
for y in range(2011,2019):
    mask_years[y] = (df_no_time['date'] >= dates_years[y]) & (df_no_time['date'] < dates_years[y+1])
df_years = {}
df_avg = {}
df_sum = {}

for y in range(2011,2019):
    df_years[y] = df_no_time[mask_years[y]]
    df_avg[y] = df_years[y].groupby(df_years[y]['date']).mean()
    df_sum[y] = df_years[y].groupby(df_years[y]['date']).sum()
for key, value in df_avg.items():
    print('Average snow depth for {0}: {1:.2f} cm'.format(key, value['ht_neige'].mean()))

years = []
cum_snow = []
for key, value in df_sum.items():
    years.append(key)
    c = value['ssfrai'].sum()
    cum_snow.append(c)
    print('Cumulated fresh snow for {0}: {1:.2f} cm'.format(key, c))

fig6 = plt.figure(num=5,figsize=(8,8),dpi=80)
plt.title('Cumulated fresh snow per year (cm)')
plt.bar(years, cum_snow)
df_sum_clusters = {}
cum_snow_clusters = {}

for key, value in dict_clusters.items():
    df_sum_clusters[key] = {}
    cum_snow_clusters[key] = []
    for y in range(2011,2019):
        #df_sum_clusters[key][y] = df_sum[y][df_sum[y]['cluster'] == key]
        df_sum_clusters[key][y] = df_years[y][df_years[y]['cluster'] == key].groupby(df_years[y]['date']).sum()
        cum_snow_clusters[key].append(df_sum_clusters[key][y]['ssfrai'].sum())
fig7 = plt.figure(num=6,figsize=(8,8),dpi=80)
plt.title('Cumulated fresh snow per year (cm)')
#for key, value in dict_clusters.items():
#    plt.plot(years, cum_snow_clusters[key])

years = [y-0.50 for y in years]
for key, value in dict_clusters.items():
    years = [y+(1/n_clusters) for y in years]
    plt.bar(years, cum_snow_clusters[key], width=1/n_clusters, label=dict_clusters[key])

plt.legend()
