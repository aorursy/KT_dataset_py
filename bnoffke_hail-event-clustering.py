import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns

import scipy as sp

from scipy import stats

from datetime import datetime as dt

from scipy.cluster.hierarchy import linkage,fcluster,dendrogram

from scipy.cluster.vq import whiten

from IPython.display import FileLink



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
hail_2016_df = pd.read_csv('/kaggle/input/noaa-severe-weather-data-inventory/swdi_hail/hail-2016.csv.gz', compression='gzip', header=0, sep=',', quotechar='"', skiprows = 2)

hail_2016_df = hail_2016_df.loc[pd.notnull(hail_2016_df['#ZTIME'])]

hail_2016_df = hail_2016_df[(hail_2016_df['MAXSIZE'] >= 0) & (hail_2016_df['PROB'] == 100)]

hail_2016_df['datetime'] = pd.to_datetime(hail_2016_df['#ZTIME'],format = '%Y%m%d%H%M%S')

print(hail_2016_df.head())

print(hail_2016_df.describe())
hail_dane_2016_df = hail_2016_df[(hail_2016_df['LAT'] <= 43.292976) & (hail_2016_df['LAT'] >= 42.846925) & (hail_2016_df['LON'] >= -89.838229) & (hail_2016_df['LON'] <= -89.012644)].reset_index()
#Use whiten as our scaling function.

hail_dane_2016_df.loc[:,'scaled_lat'] = whiten(list(hail_dane_2016_df['LAT']))

hail_dane_2016_df.loc[:,'scaled_lon'] = whiten(list(hail_dane_2016_df['LON']))

#Need to convert our timestamp to a number for clustering. This is the # of seconds since Unix time.

hail_dane_2016_df.loc[:,'raw_time'] = list(hail_dane_2016_df['datetime'].values.astype(np.int64) // 10 ** 9)

hail_dane_2016_df.loc[:,'scaled_time'] = whiten(list(hail_dane_2016_df['raw_time'])) * 100



print('Standard deviation of latitude: ' + str(np.std(hail_dane_2016_df['LAT'])))

print('Standard deviation of longitude: ' + str(np.std(hail_dane_2016_df['LON'])))

print('Standard deviation of time: ' + str(np.std(hail_dane_2016_df['raw_time'])))
#Plot histogram to get a sense of how many data points we get in a day.

ax = sns.distplot(hail_dane_2016_df['datetime'].dt.dayofyear, kde = False,bins = 365)

ax.set(xlabel = 'Day of Year', ylabel = 'Count of Events')

plt.show()
#Plot scatter plot of latitude and longitude to get a sense of hail event distribution.

sns.scatterplot(hail_dane_2016_df['LON'],hail_dane_2016_df['LAT'])
#Create our distance matrix

Z = linkage(hail_dane_2016_df[['scaled_lat','scaled_lon','scaled_time']],method = 'single',metric='euclidean')

#Create our dendrogram

dn = dendrogram(Z,p = 40, truncate_mode = 'lastp', no_labels = True)

plt.show()
#Use the whiten function to scale latitude, longitude, and time.

#  Inputs: df - Dataframe containg hail data.

#          time_scale - A number to adjust the effect time has on hail event distance.

#  Returns: df - Dataframe that additional columns for scaled values.

def scale_vals(df, time_scale = 1.0 ):

    #Use whiten as our scaling function.

    df.loc[:,'scaled_lat'] = whiten(list(df['LAT']))

    df.loc[:,'scaled_lon'] = whiten(list(df['LON']))

    #Need to convert our timestamp to a number for clustering. This is the # of seconds since Unix time.

    df.loc[:,'raw_time'] = list(df['datetime'].values.astype(np.int64) // 10 ** 9)

    df.loc[:,'scaled_time'] = whiten(list(df['raw_time'])) * time_scale

    return(df)



#Create a summary dataframe for a hail dataframe containing cluster labels.

#  Inputs: df - Dataframe containg hail data.

#  Returns: sumdf - Summary dataframe for hail storms.

def summarize_storms(df):

    sumdf = pd.DataFrame()

    nclust = max(df['label'])

    for clust in range(1,nclust + 1):

        numev = df[df['label'] == clust].shape[0]

        station = set(df['WSR_ID'][df['label'] == clust])

        strmcenlat = np.mean(df['LAT'][df['label'] == clust])

        strmcenlon = np.mean(df['LON'][df['label'] == clust])

        mindt = np.min(df['datetime'][df['label'] == clust])

        maxdt = np.max(df['datetime'][df['label'] == clust])

        maxsize = np.round(np.max(df['MAXSIZE'][df['label'] == clust]),2)

        dur = maxdt - mindt

        sumdf = sumdf.append({'label' : clust,

                              'num_events' : numev,

                              'stations' : station,

                              'storm_cen_lat' : strmcenlat,

                              'storm_cen_lon' : strmcenlon,

                              'mindt' : mindt,

                              'maxdt' : maxdt,

                              'timespan' : dur,

                              'max_size' : maxsize},

                            ignore_index = True)

    return(sumdf)



#Generates reports of cluster data, plots latitude/longitude of clustered hail events, and plots distribution of hail events in time for the cluster.

#  Input: df - Dataframe containing hail event-level data.

#         sumdf - Summary dataframe for clust-level data.

#         select - Parameter to choose specific clusters, defaults to all clusters.

#         min_num_events - Only include clusters with >= x events.

def report_storms(df, sumdf, select = 'all', min_num_events = 10):

    nlim = max(df['LAT'])

    slim = min(df['LAT'])

    wlim = min(df['LON'])

    elim = max(df['LON'])

    if select != 'all':

        rows = list(np.array(select) - 1)

        sumdf = sumdf.loc[rows,:]

    sumdf = sumdf.loc[sumdf['num_events'] >= min_num_events]

    for index,row in sumdf.iterrows():

        print('Hail Storm ' + str(row['label']))

        print('  Detecting Station(s): ' + str(row['stations']))

        print('  Storm Centroid: ' + str(row['storm_cen_lat']) + ',' + str(row['storm_cen_lon']))

        print('  Num Hail Events: ' + str(row['num_events']))

        print('  First Detection: ' + str(row['mindt']))

        print('  Latest Detection: ' + str(row['maxdt']))

        print('  Timespan: ' + str(row['timespan']))

        print ('  Largest Size: ' + str(row['max_size']) + 'in.')

        storm_rows = df[df['label'] == row['label']]

        time_dist = storm_rows['raw_time'] - np.mean(storm_rows['raw_time'])

        if row['num_events'] > 1:

            g = sns.scatterplot(data = storm_rows, x = 'LON', y= 'LAT', hue = time_dist, size = 'MAXSIZE')

            g.set(xlim = (wlim,elim), ylim = (slim,nlim))

            g.set_title('Spatial Plot of Hail Event Cluster')

            plt.show()

            sns.distplot(time_dist, bins = 10).set_title('Time Distribution of Hail Events')

            plt.show()

        else:

            g = sns.scatterplot(data = storm_rows, x = 'LON', y= 'LAT')

            g.set(xlim = (wlim,elim), ylim = (slim,nlim))

            g.set_title('Spatial Plot of Hail Event Cluster')

            

#Function to generate a dataframe containing a specificed range of cluster sizes, output a string for a calculated

#field in Tableau that can utilize a parameter to switch between cluster sets.

#  Input: df - Dataframe that will be extended to contain a range of cluster labels for varying max number of clusters

#         Z - distance matrix determined from hierarchical clustering

#         minclust - Minimum number of clusters to generate

#         maxclust - Maximum number of clusters to generate

# Notes - Consider providing a list of specific clusters sizes as an argument

def gen_tableau_clusters(df,Z,minclust,maxclust):

    clust_range = range(minclust,maxclust + 1)

    #tableau_calc_field = ''

    print('Copy the text below for the Tableau calculated field:')

    print('CASE [Cluster Set]')

    for i in clust_range:

        curr_label = 'Label' + str(i)

        df.loc[:,curr_label] = fcluster(Z, i, criterion = 'maxclust')

        print('    WHEN ' + str(i) + ' THEN [' + curr_label + ']')

    print('END')

    #hail_latesept_2016_df.to_csv('cluster_data_source.csv',index=False)

    #FileLink(r'cluster_data_source.csv')

    return(df) 
#Click the output button to the right to see a robust output of all clusters greater than 10 events.

hail_dane_2016_df.loc[:,'label'] = fcluster(Z, 28, criterion = 'maxclust')

sorted_df = summarize_storms(hail_dane_2016_df).sort_values(by = 'mindt')

report_storms(hail_dane_2016_df,sorted_df)
report_storms(hail_dane_2016_df,sorted_df,select=[26,27])
#Find events on 9/19/2016 and add scaled values

hail_latesept_2016_df = scale_vals(hail_2016_df[(hail_2016_df['datetime'].dt.year == 2016) & (hail_2016_df['datetime'].dt.month == 9) & (hail_2016_df['datetime'].dt.day == 19)], time_scale = 50)

#Calculate our distance matrix

Z = linkage(hail_latesept_2016_df[['scaled_lat','scaled_lon','scaled_time']], method = 'single', metric = 'euclidean')

#Create dendrogram

dn = dendrogram(Z, no_labels = True)

plt.show()
#Click the output button to the right to see a robust output of all clusters greater than 10 events.

hail_latesept_2016_df.loc[:,'label'] = fcluster(Z, 100, criterion = 'maxclust')

hail_storms_latesept_2016_df = summarize_storms(hail_latesept_2016_df).sort_values(by = 'mindt')

report_storms(hail_latesept_2016_df,hail_storms_latesept_2016_df)
report_storms(hail_latesept_2016_df,hail_storms_latesept_2016_df,select=[72])
tmp_df = gen_tableau_clusters(hail_latesept_2016_df,Z,1,200)
tmp_df.to_csv('hail_latesept_2016_df.csv',index=False)

FileLink(r'hail_latesept_2016_df.csv')
cluster_stats = pd.DataFrame()

max_clust_size = hail_latesept_2016_df.shape[0]

nc = range(0,hail_latesept_2016_df.shape[0])

for n in nc:

    hail_latesept_2016_df.loc[:,'tmp_label'] = fcluster(Z, n + 1, criterion = 'maxclust')

    new_max_clust_size = max(hail_latesept_2016_df.groupby('tmp_label')['tmp_label'].count())

    delta_max_clust_size = max_clust_size - new_max_clust_size

    max_clust_size = new_max_clust_size

    min_clust_size = min(hail_latesept_2016_df.groupby('tmp_label')['tmp_label'].count())

    mean_clust_size = np.mean(hail_latesept_2016_df.groupby('tmp_label')['tmp_label'].count())

    med_clust_size = np.median(hail_latesept_2016_df.groupby('tmp_label')['tmp_label'].count())

    #print(str(n+1) + ', ' + str(max_clust_size)+ ', ' + str(min_clust_size) + ', ' + str(mean_clust_size)+ ', ' + str(med_clust_size))

    cluster_stats = cluster_stats.append({'n_clust' : n + 1,

                          'max_clust_size': max_clust_size,

                          'min_clust_size': min_clust_size,

                          'mean_clust_size': mean_clust_size,

                          'med_clust_size': med_clust_size,

                          'delta_max_clust_size': delta_max_clust_size},

                        ignore_index = True)



#Drop the temporary label since it isn't useful outside of these calculations

hail_latesept_2016_df.drop(labels='tmp_label',axis=1,inplace=True)
#Sort by change in max cluster size to see which cluster numbers produce the largest change

cluster_stats.sort_values(by = 'delta_max_clust_size', ascending = False).head(20)
#Create plot of cluster summary stats

sns.lineplot(x = 'n_clust', y = 'value', hue = 'variable', data = pd.melt(cluster_stats,id_vars = 'n_clust', value_vars = ['max_clust_size','min_clust_size','mean_clust_size','med_clust_size','delta_max_clust_size']))