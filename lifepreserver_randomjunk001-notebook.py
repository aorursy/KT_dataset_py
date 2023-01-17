%matplotlib inline
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 15:02:58 2016

@author: Z
"""

north_pole = (90,0)
weight_limit = 1000
sleigh_weight = 10

import pandas as pd
import numpy as np
import time
from haversine import haversine
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

def weighted_trip_length(stops, weights): 
    tuples = [tuple(x) for x in stops.values]
    # adding the last trip back to north pole, with just the sleigh weight
    tuples.append(north_pole)
    weights.append(sleigh_weight)
    
    dist = 0.0
    prev_stop = north_pole
    prev_weight = sum(weights)
    for location, weight in zip(tuples, weights):
        dist = dist + haversine(location, prev_stop) * prev_weight
        prev_stop = location
        prev_weight = prev_weight - weight
    return dist

def weighted_reindeer_weariness(all_trips):
    uniq_trips = all_trips.TripId.unique()
    
    if any(all_trips.groupby('TripId').Weight.sum() > weight_limit):
        raise Exception("One of the sleighs over weight limit!")
 
    dist = 0.0
    for t in uniq_trips:
        this_trip = all_trips[all_trips.TripId==t]
        dist = dist + weighted_trip_length(this_trip[['Latitude','Longitude']], this_trip.Weight.tolist())
    
    return dist    
    
a = time.time()
gifts = pd.read_csv('../input/gifts.csv')

usecols = ['Latitude', 'Longitude']
#Use K-means to cluster all gifts
c = KMeans(n_clusters = 1600, random_state = 0, n_jobs = -1, tol = 1e-3)
c.fit(gifts[usecols])
gifts['cluster'] = c.predict(gifts[usecols])

num_trips = 0
w_packed = 0
#Divide gifts into groups and assign trip ids
for idx, g in gifts.groupby('cluster'):
    tripids = []
    for i, row in g.iterrows():  
        w_packed += row['Weight']
        if w_packed > 1000:
            num_trips += 1
            tripids.append(num_trips)
            w_packed = row['Weight']
        else:
            #Append trip
            tripids.append(num_trips)
            
    gifts.ix[gifts['cluster'] == idx, 'TripId'] = tripids
    num_trips += 1

score = weighted_reindeer_weariness(gifts)

print('Cur  :', score)
print('Top1 : 12389228461.1')
print('%diff : ', (12389228461.1 - score) / 12389228461.1 * 100)
print('Timeused :', time.time() - a)
gifts.groupby('TripId').Weight.sum().hist(bins = 100)
gifts.TripId.unique().shape
def visualize_cluster(gifts, clusters = None):
    if clusters == None:
        try:
            clusters = gifts.cluster.unique()
        except AttributeError as e:
            raise (e, 'No Cluster value in gifts')
    for c in clusters:
        y = gifts.Latitude[gifts.cluster == c]
        x = gifts.Longitude[gifts.cluster == c]
        plt.scatter(x, y)
        plt.xlim([-180,180])
        plt.ylim([-90, 90])
visualize_cluster(gifts, clusters = [1])
visualize_cluster(gifts, clusters = [i for i in range(100)])
visualize_cluster(gifts, clusters = [i for i in range(1000)])
gifts = pd.read_csv('../input/gifts.csv')
gifts['cluster'] = c.predict(gifts[usecols])

num_trips = 0
w_packed = 0
#Divide gifts into groups and assign trip ids
for idx, g in gifts.groupby('cluster'):
    tripids = []
    for i, row in g.iterrows():  
        w_packed += row['Weight']
        if w_packed > 1000:
            num_trips += 1
            tripids.append(num_trips)
            w_packed = row['Weight']
        else:
            #Append trip
            tripids.append(num_trips)
            
    gifts.ix[gifts['cluster'] == idx, 'TripId'] = tripids
#    num_trips += 1

score = weighted_reindeer_weariness(gifts)

print('Cur  :', score)
print('Top1 : 12389228461.1')
print('%diff : ', (12389228461.1 - score) / 12389228461.1 * 100)
print('Timeused :', time.clock() - a)
num_trips = 0
w_packed = 0
idx = 0
#Divide gifts into groups and assign trip ids
clusters = set(gifts['cluster'].unique())
distances = squareform(pdist(c.cluster_centers_))
distances[distances == 0] = 9999

while clusters:
    g = gifts[gifts['cluster'] == idx]
    tripids = []
    for i, row in g.iterrows():  
        w_packed += row['Weight']
        if w_packed > 1000:
            num_trips += 1
            tripids.append(num_trips)
            w_packed = row['Weight']
        else:
            #Append trip
            tripids.append(num_trips)
            
    gifts.ix[gifts['cluster'] == idx, 'TripId'] = tripids
#    num_trips += 1
    remain = list(clusters)
    nearest_cluster = remain[np.argmin(distances[idx, remain])]
    idx = nearest_cluster
    clusters.remove(idx)

score = weighted_reindeer_weariness(gifts)

print('Cur  :', score)
print('Top1 : 12389228461.1')
print('%diff : ', (12389228461.1 - score) / 12389228461.1 * 100)
print('Timeused :', time.clock() - a)
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
