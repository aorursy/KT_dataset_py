import numpy as np

import pandas as pd

import random

import os

import matplotlib.pyplot as plt

import seaborn as sns 

import math

import netCDF4

import gc

from scipy.interpolate import griddata

from osgeo import gdal

from mpl_toolkits.basemap import Basemap

from sklearn.preprocessing import StandardScaler

from math import cos, asin, sqrt

from numpy import nansum,nanmean,linspace,meshgrid

from numpy import meshgrid

from sklearn.cluster import DBSCAN

from matplotlib.colors import Normalize

%matplotlib inline



Random_seed=123
def distance(lat1, lon1, lat2, lon2):

    p = 0.017453292519943295

    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p)*cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2

    return 12742 * asin(sqrt(a))



def round_up_to_even(f):

    return math.ceil(f / 2.) * 2



n = 11061987

s = 11061987

skip = sorted(random.sample(range(n),n-s))

        

df_path = "../input/geonames-database/geonames.csv"

df = pd.read_csv(df_path,skiprows=skip,index_col="geonameid")

ISO = pd.read_csv('../input/alpha-country-codes/Alpha__2_and_3_country_codes.csv', sep=';')

Population = pd.read_csv('../input/population-by-country-2020/population_by_country_2020.csv')



C = (df.dtypes == 'object')

CategoricalVariables = list(C[C].index)

Integer = (df.dtypes == 'int64') 

Float   = (df.dtypes == 'float64') 

NumericVariables = list(Integer[Integer].index) + list(Float[Float].index)



df=df.drop(['alternatenames','admin2 code','admin3 code','admin4 code','cc2'], axis=1)



df["latitude_app"] = df.apply(lambda row: round_up_to_even(row['latitude']),axis=1)

df["longitude_app"] = df.apply(lambda row: round_up_to_even(row['longitude']),axis=1)



elevation_table = df[['elevation','latitude_app','longitude_app']].groupby(['latitude_app',

                'longitude_app']).agg({'elevation': lambda x: x.mean(skipna=True)}).sort_values(by=['latitude_app', 

                'longitude_app'], ascending=False).reset_index()



df = pd.merge(df,  elevation_table,  on =['latitude_app', 'longitude_app'],  how ='inner')



WorldAverageElevation = 840

df['elevation_y']=df['elevation_y'].fillna(WorldAverageElevation)



df=df.drop(['elevation_x'], axis=1)



ISO['Country'] = ISO.apply(lambda row: str.rstrip(row['Country']),axis=1)



ISO_toMerge = ISO.drop(['Alpha-3 code','Numeric'], axis=1)

ISO_toMerge=ISO_toMerge.rename(columns={"Alpha-2 code": "country code"})

df = pd.merge(df, ISO_toMerge,  on ='country code',  how ='inner')



df_sample = df.sample(n=100000,random_state=Random_seed)



Aggregated = df[['name','Country']]

Aggregated = Aggregated.groupby(['Country']).agg(['count']).sort_values([('name', 'count')], ascending=False)

Aggregated['Percentage'] = round(Aggregated[['name']] / df.shape[0],2)

Aggregated.columns = Aggregated.columns.get_level_values(0)

Aggregated.columns = [''.join(col).strip() for col in Aggregated.columns.values]



Population = Population.rename(columns={"Country (or dependency)": "Country"})



Population[['Country']] = Population[['Country']].replace("Czech Republic (Czechia)", "Czechia")

Population[['Country']] = Population[['Country']].replace("United States", "United States of America")

Population[['Country']] = Population[['Country']].replace("United Kingdom", "United Kingdom of Great Britain and Northern Ireland")

Population[['Country']] = Population[['Country']].replace("Vietnam", "Viet Nam")

Population[['Country']] = Population[['Country']].replace("Laos", "Lao People Democratic Republic")

Population[['Country']] = Population[['Country']].replace("State of Palestine", "Palestine")

Population[['Country']] = Population[['Country']].replace("North Macedonia", "Republic of North Macedonia")

Population[['Country']] = Population[['Country']].replace("Russia", "Russian Federation")

Population[['Country']] = Population[['Country']].replace("Syria", "Syrian Arab Republic")



Sample_Size = 1000000



Population_Merged = pd.merge(ISO_toMerge,Population,  on ='Country',  how ='inner')



Population_Merged[['Population Perc']] = Population_Merged[['Population (2020)']]/Population_Merged[['Population (2020)']].sum()



Population_Merged = pd.merge(Population_Merged,Aggregated,  on ='Country',  how ='inner')



Population_Merged[['Sample size']] = Population_Merged['Population Perc']  / Population_Merged['name']*Population_Merged['name'].sum()*Sample_Size



Population_toMerge = Population_Merged.loc[:, Population_Merged.columns.intersection(['Country','Sample size'])]



df = pd.merge(df,Population_toMerge,  on ='Country',  how ='inner')



Total_Probability = df[['Sample size']].sum()



df[['Sample size']] = df[['Sample size']] / Total_Probability



vec = df[['Sample size']]



df_sampled = df.sample(n=Sample_Size, weights='Sample size',random_state=Random_seed)



gc.collect()
df_countries = df[df['feature code']=="PCLI"]

df_admin = df[df['feature code']=="ADM1"]

df_regions = df[df['feature code']=="RGN"]

df_continents = df[df['feature code']=="CONT"]

df_nature = df[df['feature code']=="AREA"]



df_administrative = df[df['feature class']=="A"]

df_hydrographic = df[df['feature class']=="H"]

df_area = df[df['feature class']=="L"]

df_populated = df[df['feature class']=="P"]

df_road = df[df['feature class']=="R"]

df_spot = df[df['feature class']=="S"]

df_administrative = df[df['feature class']=="T"]

df_undersea = df[df['feature class']=="U"]

df_vegetation = df[df['feature class']=="V"]



df['feature class'].value_counts()
df_ten_and_more = df_populated[df.population>10000000]

df_five_ten = df_populated[(df.population<10000000) & (df.population>5000000)]

df_one_five = df_populated[(df.population<5000000) & (df.population>1000000)]

df_hundred_ones = df_populated[(df.population<1000000) & (df.population>100000)]
plt.figure(1, figsize=(15,8))

m1 = Basemap(projection='merc',llcrnrlat=-60,urcrnrlat=65,llcrnrlon=-180,urcrnrlon=180,

             lat_ts=0,resolution='c')



m1.fillcontinents(color='#191919',lake_color='white') 

m1.drawmapboundary(fill_color='white')                

m1.drawcountries(linewidth=0.3, color="w")              



# Plot the data

mxy_4 = m1(df_hundred_ones["longitude"].tolist(), df_hundred_ones["latitude"].tolist())

A = m1.scatter(mxy_4[0], mxy_4[1], c="blue", lw=0.001, alpha=0.8, zorder=5, marker='*')

A



mxy_3 = m1(df_one_five["longitude"].tolist(), df_one_five["latitude"].tolist())

B = m1.scatter(mxy_3[0], mxy_3[1], c="green", lw=0.1, alpha=1, zorder=5, marker='o')

B



mxy_2 = m1(df_five_ten["longitude"].tolist(), df_five_ten["latitude"].tolist())

C = m1.scatter(mxy_2[0], mxy_2[1], c="yellow", lw=0.3, alpha=1, zorder=5, marker='p')

C



mxy_1 = m1(df_ten_and_more["longitude"].tolist(), df_ten_and_more["latitude"].tolist())

D = m1.scatter(mxy_1[0], mxy_1[1], c="red", lw=0.4, alpha=1, zorder=5, marker='s')

D



plt.legend((D,C,B,A),('>10mln','5mln-10mln', '1mln-5mln', '100k-1mln'),numpoints=1, loc=1)



plt.title("Big cities of the world")

plt.show()
del df

del Population_Merged

del Aggregated

gc.collect()
df_limited = df_sampled[['longitude','latitude']].sample(n=50000,random_state=Random_seed)

df_DBSCAN = StandardScaler().fit_transform(df_limited)



DBcluster= DBSCAN(min_samples = 200,eps=0.4)

DBcluster_fit = DBcluster.fit(df_DBSCAN)

core_samples_mask = np.zeros_like(DBcluster_fit.labels_, dtype=bool)

core_samples_mask[DBcluster_fit.core_sample_indices_] = True



DBlabels = DBcluster_fit.labels_ 

DB_n_clusters_ = len(set(DBlabels)) 

DB_n_noise_ = list(DBlabels).count(-1)



print('Estimated number of clusters: %d' % DB_n_clusters_)

print('Estimated number of noise points: %d' % DB_n_noise_)
df_limited = df_limited.reset_index()

df_cluster = pd.DataFrame(DBlabels,columns=['Cluster'])



df_limited = pd.concat([df_limited, df_cluster.reindex(df_limited.index)], axis=1)
plt.figure(1, figsize=(15,8))

m5 = Basemap(projection='merc',llcrnrlat=-60,urcrnrlat=65,llcrnrlon=-180,urcrnrlon=180,

             lat_ts=0,resolution='c')



m5.fillcontinents(color='#191919',lake_color='white') 

m5.drawmapboundary(fill_color='white')                

m5.drawcountries(linewidth=0.3, color="w")



m5.drawparallels(

    np.arange(-60, 65, 2.),

    color = 'black', linewidth = 0.2,

    labels=[False, False, False, False])

m5.drawmeridians(

    np.arange(-180, 180, 2.),

    color = '0.25', linewidth = 0.2,

    labels=[False, False, False, False])



m5.drawmapscale(150., -50., -10, -55,2500,units='km', fontsize=10,yoffset=None,barstyle='fancy', labelstyle='simple',

    fillcolor1='w', fillcolor2='#000000',fontcolor='#000000',zorder=5)



unique_labels = set(DBlabels)

colors = [plt.cm.Spectral(each)

          for each in np.linspace(0, 1, len(unique_labels))]

          

core_samples_mask = np.zeros_like(DBlabels, dtype=bool)

core_samples_mask[DBcluster.core_sample_indices_] = True



# Plot the data

for k, col in zip(unique_labels, colors):

    if k == -1:

        # Black used for noise.

        col = [0, 0, 0, 1]

          

    class_member_mask = (DBlabels == k)



    xy = df_limited[class_member_mask & core_samples_mask]

    mxy = m5(xy["longitude"].tolist(), xy["latitude"].tolist())

    m5.scatter(mxy[0], mxy[1], c=tuple(col), lw=0.3, alpha=1, zorder=5, marker='p')



plt.title("Continents according to very strict DBSCAN with Euclidean metric and minimal sample: 300")

plt.show()
df_limited = df_sampled[['longitude','latitude']].sample(n=50000,random_state=Random_seed)

df_DBSCAN = StandardScaler().fit_transform(df_limited)



DBcluster= DBSCAN(eps=0.15,min_samples = 170,metric="canberra")



DBcluster_fit = DBcluster.fit(df_DBSCAN)

core_samples_mask = np.zeros_like(DBcluster_fit.labels_, dtype=bool)

core_samples_mask[DBcluster_fit.core_sample_indices_] = True



DBlabels = DBcluster_fit.labels_ 

DB_n_clusters_ = len(set(DBlabels)) 

DB_n_noise_ = list(DBlabels).count(-1)



df_limited = df_limited.reset_index()

df_cluster = pd.DataFrame(DBlabels,columns=['Cluster'])

df_limited = pd.concat([df_limited, df_cluster.reindex(df_limited.index)], axis=1)



plt.figure(1, figsize=(15,8))

m5 = Basemap(projection='merc',llcrnrlat=-60,urcrnrlat=65,llcrnrlon=-180,urcrnrlon=180,

             lat_ts=0,resolution='c')



m5.fillcontinents(color='#191919',lake_color='white') 

m5.drawmapboundary(fill_color='white')                

m5.drawcountries(linewidth=0.3, color="w")



m5.drawparallels(

    np.arange(-60, 65, 2.),

    color = 'black', linewidth = 0.2,

    labels=[False, False, False, False])

m5.drawmeridians(

    np.arange(-180, 180, 2.),

    color = '0.25', linewidth = 0.2,

    labels=[False, False, False, False])



m5.drawmapscale(150., -50., -10, -55,2500,units='km', fontsize=10,yoffset=None,barstyle='fancy', labelstyle='simple',

    fillcolor1='w', fillcolor2='#000000',fontcolor='#000000',zorder=5)



unique_labels = set(DBlabels)

colors = [plt.cm.Spectral(each)

          for each in np.linspace(0, 1, len(unique_labels))]

          

core_samples_mask = np.zeros_like(DBlabels, dtype=bool)

core_samples_mask[DBcluster.core_sample_indices_] = True



# Plot the data

for k, col in zip(unique_labels, colors):

    if k == -1:

        # Black used for noise.

        col = [0, 0, 0, 1]

          

    class_member_mask = (DBlabels == k)



    xy = df_limited[class_member_mask & core_samples_mask]

    mxy = m5(xy["longitude"].tolist(), xy["latitude"].tolist())

    m5.scatter(mxy[0], mxy[1], c=tuple(col), lw=0.3, alpha=1, zorder=5, marker='p')

    

plt.title("Continents according to DBSCAN with canberra metric")

plt.show()
df_limited = df_sampled[['longitude','latitude']].sample(n=50000,random_state=Random_seed)

df_DBSCAN = StandardScaler().fit_transform(df_limited)



DBcluster= DBSCAN(metric="l2",eps=0.14,min_samples = 120)



DBcluster_fit = DBcluster.fit(df_DBSCAN)

core_samples_mask = np.zeros_like(DBcluster_fit.labels_, dtype=bool)

core_samples_mask[DBcluster_fit.core_sample_indices_] = True



DBlabels = DBcluster_fit.labels_ 

DB_n_clusters_ = len(set(DBlabels)) 

DB_n_noise_ = list(DBlabels).count(-1)



df_limited = df_limited.reset_index()

df_cluster = pd.DataFrame(DBlabels,columns=['Cluster'])

df_limited = pd.concat([df_limited, df_cluster.reindex(df_limited.index)], axis=1)



plt.figure(1, figsize=(15,8))

m5 = Basemap(projection='merc',llcrnrlat=-60,urcrnrlat=65,llcrnrlon=-180,urcrnrlon=180,

             lat_ts=0,resolution='c')



m5.fillcontinents(color='#191919',lake_color='white') 

m5.drawmapboundary(fill_color='white')                

m5.drawcountries(linewidth=0.3, color="w")              



unique_labels = set(DBlabels)

colors = [plt.cm.Spectral(each)

          for each in np.linspace(0, 1, len(unique_labels))]

          

core_samples_mask = np.zeros_like(DBlabels, dtype=bool)

core_samples_mask[DBcluster.core_sample_indices_] = True



m5.drawparallels(

    np.arange(-60, 65, 2.),

    color = 'black', linewidth = 0.2,

    labels=[False, False, False, False])

m5.drawmeridians(

    np.arange(-180, 180, 2.),

    color = '0.25', linewidth = 0.2,

    labels=[False, False, False, False])



m5.drawmapscale(150., -50., -10, -55,2500,units='km', fontsize=10,yoffset=None,barstyle='fancy', labelstyle='simple',

    fillcolor1='w', fillcolor2='#000000',fontcolor='#000000',zorder=5)



# Plot the data

for k, col in zip(unique_labels, colors):

    if k == -1:

        # Black used for noise.

        col = [0, 0, 0, 1]

          

    class_member_mask = (DBlabels == k)



    xy = df_limited[class_member_mask & core_samples_mask]

    mxy = m5(xy["longitude"].tolist(), xy["latitude"].tolist())

    m5.scatter(mxy[0], mxy[1], c=tuple(col), lw=0.3, alpha=1, zorder=5, marker='p')

    

plt.title("Continents according to DBSCAN with L2 metric")

plt.show()
norm = Normalize()



df_map = df_sampled.sample(n=10000, random_state=Random_seed)



plt.figure(1, figsize=(15,8))

m2 = Basemap(projection='merc',llcrnrlat=-60,urcrnrlat=65,llcrnrlon=-180,urcrnrlon=180,lat_ts=0,resolution='c')



m2.fillcontinents(color='#C0C0C0', lake_color='#7093DB') 

m2.drawmapboundary(fill_color='white')                

m2.drawcountries(linewidth=.75, linestyle='solid', color='#000073',antialiased=True,zorder=3) 



m2.drawparallels(

    np.arange(-60, 65, 2.),

    color = 'black', linewidth = 0.2,

    labels=[False, False, False, False])

m2.drawmeridians(

    np.arange(-180, 180, 2.),

    color = '0.25', linewidth = 0.2,

    labels=[False, False, False, False])



x,y = m2(*(df_map.longitude.values, df_map.latitude.values))



numcols, numrows = 1000, 1000

xi = np.linspace(x.min(), x.max(), numcols)

yi = np.linspace(y.min(), y.max(), numrows)

xi, yi = np.meshgrid(xi, yi)



x_1, y_1, z_1 = df_map.longitude.values, df_map.latitude.values, df_map.elevation_y.values

zi = griddata((x,y),df_map.elevation_y.values,(xi, yi),method='linear')



con = m2.contourf(xi, yi, zi, zorder=4, alpha=0.7, cmap='RdPu')



m2.scatter(x,y,color='#545454',edgecolor='#ffffff',alpha=.75,s=50 * norm(df_map.elevation_y),cmap='RdPu',vmin=zi.min(), vmax=zi.max(), zorder=4)



m2.drawmapscale(150., -50., -10, -55,2500,units='km', fontsize=10,yoffset=None,barstyle='fancy', labelstyle='simple',

    fillcolor1='w', fillcolor2='#000000',fontcolor='#000000',zorder=5)



cbar = plt.colorbar(con, orientation='horizontal', fraction=.057, pad=0.05)

cbar.set_label("Regional elevation - m")



plt.title("Elevation of the region with mountains marked")

plt.show()