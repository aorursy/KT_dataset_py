# BigQuery works automatically on kaggle.  Need to signup and authenticate with google if used on local machine

from google.cloud import bigquery

import bq_helper



%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import numpy as np

from matplotlib.pyplot import figure

from bq_helper import BigQueryHelper

import pandas as pd

from sklearn.cluster import DBSCAN

from sklearn.neighbors import NearestNeighbors



# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package

chicago_crime = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                   dataset_name="chicago_crime")
bq_assistant = BigQueryHelper("bigquery-public-data", "chicago_crime")

#List all availablt tables

bq_assistant.list_tables()
# First couple rows of the crime table

bq_assistant.head("crime", num_rows=5)
# Schema and dtypes of the variables in the crime table

bq_assistant.table_schema("crime")
query_all_entries = "SELECT COUNT(*) FROM `bigquery-public-data.chicago_crime.crime`"

bq_assistant.query_to_pandas(query_all_entries)
query_null_coords = "SELECT COUNT(*) FROM `bigquery-public-data.chicago_crime.crime` WHERE x_coordinate IS NULL"

bq_assistant.query_to_pandas(query_null_coords)
null_percentage = (63640 / 6846406) * 100

print(str(null_percentage) + " %")
crime_types_sql = """

                SELECT DISTINCT primary_type

                FROM `bigquery-public-data.chicago_crime.crime`

                ORDER BY primary_type

"""

bq_assistant.query_to_pandas(crime_types_sql)
sql = """

    SELECT COUNT(*)

    FROM `bigquery-public-data.chicago_crime.crime`

    WHERE (latitude IS NOT NULL) AND (latitude > 39)

    AND date >= '2018-01-01 00:00:00'

    AND date <= '2019-01-01 00:00:00' 

    AND (primary_type = 'ASSAULT' 

    OR primary_type = 'BATTERY'

    OR primary_type = 'HOMICIDE'

    OR primary_type = 'KIDNAPPING'

    OR primary_type = 'ROBBERY')

"""

#df = client.query(sql).to_dataframe()

#df.sample(n=3, random_state=1)



bq_assistant.estimate_query_size(sql)

bq_assistant.query_to_pandas(sql)
#for DBSCAN

violent_crime_sql = """

    SELECT DISTINCT latitude, longitude 

    FROM `bigquery-public-data.chicago_crime.crime`

    WHERE (latitude IS NOT NULL) AND (latitude > 39)

    AND date >= '2018-01-01 00:00:00'

    AND date <= '2019-01-01 00:00:00' 

    AND (primary_type = 'ASSAULT'

    OR primary_type = 'BATTERY'

    OR primary_type = 'HOMICIDE'

    OR primary_type = 'KIDNAPPING'

    OR primary_type = 'ROBBERY')

"""



df = bq_assistant.query_to_pandas(violent_crime_sql)

df.to_csv('crime_coords.csv', index=False, chunksize=1)

coords = df.as_matrix(columns=['latitude', 'longitude'])



#type(df) <- pandas.core.frame.DataFrame
#for OPTICS

violent_crime_xy_sql = """

    SELECT DISTINCT x_coordinate, y_coordinate 

    FROM `bigquery-public-data.chicago_crime.crime`

    WHERE (latitude IS NOT NULL) AND (latitude > 39)

    AND date >= '2018-01-01 00:00:00'

    AND date <= '2019-01-01 00:00:00' 

    AND (primary_type = 'ASSAULT'

    OR primary_type = 'BATTERY'

    OR primary_type = 'HOMICIDE'

    OR primary_type = 'KIDNAPPING'

    OR primary_type = 'ROBBERY')

"""



df_xy = bq_assistant.query_to_pandas(violent_crime_xy_sql)

df_xy.sample(40000).to_csv('crime_xy_coords.csv', index=False, chunksize=1)

coords_xy = df_xy.as_matrix(columns=['x_coordinate', 'y_coordinate'])
#lat/lon array for DBSCAN

print(coords)



#x,y dataframe for optics

print(coords_xy)
#plt.scatter('latitude', 'longitude', color='red', s=0.01, data=df)

#plt.xlabel('longitude')

#plt.ylabel('latitude')

#plt.show()



from mpl_toolkits.basemap import Basemap



fig, ax = plt.subplots(figsize = (12, 12))



#m = Basemap(projection='gnom', resolution='l', epsg=26971, llcrnrlon=-87.95, llcrnrlat=41.60, urcrnrlon=-87.50, urcrnrlat=42.05)

m = Basemap(llcrnrlon=-87.95,

            llcrnrlat=41.60, 

            urcrnrlon=-87.50,

            urcrnrlat=42.05, 

            projection='merc', 

            resolution='l', 

            lon_0=-87.725,

            lat_0=41.825,

            epsg = 4269)



m.arcgisimage(service='World_Street_Map', xpixels=1000, verbose=False)

x, y = m(coords[:,1], coords[:,0])

m.scatter(x, y, color='red', s=0.1, data=df)

plt.show()
#this is for DBSCAN using lat/lon

X = df.values



#finding 4th nearest neighbor

nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(X)

distances, indices = nbrs.kneighbors(X)



#print(nbrs.kneighbors(X))

print(indices[:10])

print(distances[:10])
#using x,y coordinates for OPTICS

X_1 = df_xy.values

nbrs_1 = NearestNeighbors(n_neighbors=4, algorithm='auto', metric='euclidean').fit(X_1)

distances_1, indices_1 = nbrs_1.kneighbors(X_1)



#print(nbrs.kneighbors(X))

print(indices_1)

print(distances_1)
a = distances[:]

dist_sorted = a[a[:,3].argsort()[::-1]][:,3]



b = indices[:]

indices_sorted = np.array([i for i in range(len(dist_sorted))])



print(indices_sorted[:20])

print(dist_sorted[:20])
a_1 = distances_1[:]

dist_sorted_1 = a_1[a_1[:,3].argsort()[::-1]][:,3]



b_1 = indices_1[:]

indices_sorted_1 = np.array([i for i in range(len(dist_sorted_1))])



print(indices_sorted_1[:20])

print(dist_sorted_1[:20])
plt.figure(figsize=(12,8))

plt.plot(indices_sorted_1, dist_sorted_1)

plt.xlabel('point indices')

plt.ylabel('distance from 4th nearest neighbor')

plt.vlines(50, ymin=0, ymax=6000, linestyles='dashed')

plt.show()
#Can change index to see corresponding epsilon value

print("Epsilon at index 100: ",dist_sorted_1[100])
from sklearn import metrics



kms_per_radian = 6371.0088

epsilon = 0.5 / kms_per_radian



db = DBSCAN(eps=epsilon, min_samples=70, algorithm='ball_tree', metric='haversine').fit(coords)



cluster_labels = db.labels_

n_clusters = len(set(cluster_labels))



clusters = pd.Series([coords[cluster_labels == n] for n in range(-1, n_clusters)])

print('Number of clusters: {}'.format(n_clusters))
import matplotlib.cm as cmx

import matplotlib.colors as colors



# define a helper function to get the colors for different clusters

def get_cmap(N):

    '''

    Returns a function that maps each index in 0, 1, ... N-1 to a distinct 

    RGB color.

    '''

    color_norm  = colors.Normalize(vmin=0, vmax=N-1)

    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='nipy_spectral') 

    def map_index_to_rgb_color(index):

        return scalar_map.to_rgba(index)

    return map_index_to_rgb_color
print(clusters)
from mpl_toolkits.basemap import Basemap



fig, ax = plt.subplots(figsize = (12, 12))



m = Basemap(llcrnrlon=-87.95,

            llcrnrlat=41.60, 

            urcrnrlon=-87.50,

            urcrnrlat=42.05, 

            projection='merc', 

            resolution='l', 

            lon_0=-87.725,

            lat_0=41.825,

            epsg = 4269)



m.arcgisimage(service='World_Street_Map', xpixels=2000, verbose=False)



unique_label = np.unique(cluster_labels)



# get different color for different cluster

cmaps = get_cmap(n_clusters)



# plot different clusters on map, note that the black dots are 

# outliers that not belone to any cluster. 

for i, cluster in enumerate(clusters):

    lons_select = cluster[:, 1]

    lats_select = cluster[:, 0]

    x, y = m(lons_select, lats_select)

    m.scatter(x, y, 0.15, marker='o', color=cmaps(i), zorder=10)



plt.show()
# THIS DOESNT WORK ON KAGGLE, BUT WORKS ON MY LAPTOP.  MUST 'pip3 install pyclustering'

# FOR THIS TO WORK HOWEVER

import random;



from pyclustering.cluster import cluster_visualizer;

from pyclustering.cluster.optics import optics, ordering_analyser, ordering_visualizer;



from pyclustering.utils import read_sample, timedcall;



from pyclustering.samples.definitions import SIMPLE_SAMPLES, FCPS_SAMPLES;





def template_clustering(path_sample, eps, minpts, amount_clusters = None, visualize = True, ccore = False):

    sample = read_sample(path_sample);

    #yeah = sample

    

    optics_instance = optics(sample, eps, minpts, amount_clusters, ccore).process()

    #(ticks, _) = timedcall(optics_instance.process);

    

    #print("Sample: ", path_sample, "\t\tExecution time: ", ticks, "\n")

    

    if (visualize is True):

        clusters = optics_instance.get_clusters()

        noise = optics_instance.get_noise()

        print(optics_instance)

    

        visualizer = cluster_visualizer()

        visualizer.append_clusters(clusters, sample)

        visualizer.append_cluster(noise, sample, marker = 'x')

        visualizer.show()

    

        ordering = optics_instance.get_ordering()

        analyser = ordering_analyser(ordering)

        

        ordering_visualizer.show_ordering_diagram(analyser, amount_clusters)

    

coords_list = coords.tolist()

nonneg_coords = [[x, -y] for x, y in coords_list]

#print(nonneg_coords)



eps = 5



def cluster_crime():

    template_clustering(nonneg_coords, 0.5, 4)   

    

def cluster_sample1():

    template_clustering(SIMPLE_SAMPLES.SAMPLE_SIMPLE1, 0.5, 3);



#cluster_sample1()

#cluster_crime()

#print(SIMPLE_SAMPLES.SAMPLE_SIMPLE1)