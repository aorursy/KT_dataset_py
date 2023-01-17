# This Python 3 environment comes with many helpful analytics libraries installed



# import libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from sklearn.datasets.samples_generator import make_blobs

%matplotlib inline

# create dataset

np.random.seed(0)
#select center points for cluters of randomly generated numbers

center_pts = [[4,5],[-3,-2], [2,-3], [1,1]]
# make_blobs

X,y = make_blobs(n_samples = 5000, centers = center_pts , cluster_std = 0.9 )
#scatter plot of randomlhy generated data

plt.scatter(X[:,0],X[:,1], marker = '.')
#initialize KMeans with parameters

k_means = KMeans(init = 'k-means++', n_clusters = 4, n_init = 12)
#model with KMeans

k_means.fit(X)

k_means_labels = k_means.labels_

k_means_labels
# get the location of the cluster centers using 

# KMeans . cluster_centers



k_means_cluster_centers = k_means.cluster_centers_

k_means_cluster_centers
#visualize data 

def KM_plot(center_pts, k_means_labels, k_means_cluster_centers):

    #initialize the plot with specific dimensions

    fig = plt.figure(figsize = (6,4))



    #set up color map using unique labels in k_means_labels

    colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))



    #create plot

    ax = fig.add_subplot(1,1,1)



    #for loop that plots the data points and centroids

    for k , col in zip(range(len(center_pts)),colors):

        #creat list of all data points in the cluter that are labelled true

        my_members = (k_means_labels == k)



        #define the centroid

        cluster_center = k_means_cluster_centers[k]



        #plot data points with color col

        ax.plot(X[my_members, 0], X[my_members,1], 'w', markerfacecolor = col, marker = '.')



        #plot centroids with color col and darker outline

        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor = col, markeredgecolor = 'k')



    #plot title

    ax.set_title('KMeans')



    #show plot

    plt.show()    
KM_plot(center_pts, k_means_labels, k_means_cluster_centers)