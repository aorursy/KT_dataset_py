import sys # for using sys.maxint

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf # tensorflow for all the ML magic



%matplotlib inline

import matplotlib.pyplot as plt # for plotting graphs

import seaborn as sns # for prettier graphs

sns.set(style="dark")
mean_01 = np.array([0.0, 0.0]) 

cov_01 = np.array([[1, 0.3], [0.3, 1]]) 

dist_01 = np.random.multivariate_normal(mean_01, cov_01, 100) 

   

mean_02 = np.array([6.0, 7.0]) 

cov_02 = np.array([[1.5, 0.3], [0.3, 1]]) 

dist_02 = np.random.multivariate_normal(mean_02, cov_02, 100) 

   

mean_03 = np.array([7.0, -5.0]) 

cov_03 = np.array([[1.2, 0.5], [0.5, 1,3]]) 

dist_03 = np.random.multivariate_normal(mean_03, cov_01, 100) 

   

mean_04 = np.array([2.0, -7.0]) 

cov_04 = np.array([[1.2, 0.5], [0.5, 1,3]]) 

dist_04 = np.random.multivariate_normal(mean_04, cov_01, 100) 

   

dummy_data = np.vstack((dist_01, dist_02, dist_03, dist_04)) 

np.random.shuffle(dummy_data) 
def plot(data, centroids = np.array([]), type="no-track"): 

    plt.figure(figsize=(10, 10))

    plt.scatter(data[:, 0], data[:, 1], marker = '.',  

                color = 'gray', label = 'data points') 

    if len(centroids) != 0:

        if type == "track":

            plt.scatter(centroids[:-1, 0], centroids[:-1, 1],  

                    color = 'black', label = 'previously selected centroids') 

            plt.scatter(centroids[-1, 0], centroids[-1, 1],  

                    color = 'red', label = 'next centroid') 

            plt.title('Select % d th centroid'%(centroids.shape[0]))

        else:

            plt.scatter(centroids[:, 0], centroids[:, 1],  

                    color = 'red', label = 'centroid') 

    

    plt.legend() 

    plt.xlim(-5, 12) 

    plt.ylim(-10, 15) 

    plt.show() 
plot(dummy_data) 
# utility to assign centroids to data points

def assignCentroids(X, C):  

    expanded_vectors = tf.expand_dims(X, 0)

    expanded_centroids = tf.expand_dims(C, 1)

    distance = tf.math.reduce_sum( tf.math.square( tf.math.subtract( expanded_vectors, expanded_centroids ) ), axis=2 )

    return tf.math.argmin(distance, 0)

                                              

# utility to recalculate centroids

def reCalculateCentroids(X, X_labels):

    sums = tf.math.unsorted_segment_sum( X, X_labels, k )

    counts = tf.math.unsorted_segment_sum( tf.ones_like( X ), X_labels, k  )

    return tf.math.divide( sums, counts )
# Number of cluster we wish to divide the data into( user tunable )

k = 4



# Max number of allowed iterations for the algorithm( user tunable )

epochs = 10000



data = pd.DataFrame(dummy_data)



X = tf.Variable(data.values, name="X")

X_labels = tf.Variable(tf.zeros(shape=(X.shape[0], 1)),name="C_lables")

C = tf.Variable(data.sample(k).values, name="C")



for epoch in range( epochs ):

    X_labels =  assignCentroids( X, C )

    C = reCalculateCentroids( X, X_labels )
plot(dummy_data, C.numpy(), "no-track")
# return an np-array of k points based on k-means++

def selectCentroids(k, dummy_data):

    centroids = [] 

    

    # pick the first centroid at random

    centroids.append(dummy_data[np.random.randint( 

            dummy_data.shape[0]), :]) 

    

    plot(dummy_data, np.array(centroids), "track") 

    

    # compute remaining k - 1 centroids 

    for centroid_index in range(k - 1):  

        # points from nearest centroid 

        distance_array = [] 

        

        # iterate over the data points for each centroid

        # to find the distance from nearest chosen centroids

        for i in range(dummy_data.shape[0]): 

            point = dummy_data[i, :]

            distance = sys.maxsize 

              

            ## compute distance of 'point' from each of the previously 

            ## selected centroid and store the minimum distance 

            for j in range(len(centroids)): 

                temp_distance = np.linalg.norm(point - centroids[j]) 

                distance = min(distance, temp_distance) 

            distance_array.append(distance) 

              

        ## select data point with maximum distance as our next centroid 

        distance_array = np.array(distance_array) 

        next_centroid = dummy_data[np.argmax(distance_array), :]

        centroids.append(next_centroid) 



        plot(dummy_data, np.array(centroids), "track") 

    return np.array(centroids)
# Number of cluster we wish to divide the data into( user tunable )

k = 4



# Max number of allowed iterations for the algorithm( user tunable )

epochs = 10000



data = pd.DataFrame(dummy_data)



X = tf.Variable(data.values, name="X")

X_labels = tf.Variable(tf.zeros(shape=(X.shape[0], 1)),name="C_lables")

C = tf.Variable(selectCentroids(k, dummy_data), name="C")



for epoch in range( epochs ):

    X_labels =  assignCentroids( X, C )

    C = reCalculateCentroids( X, X_labels )

    

plot(dummy_data, C.numpy(), "no-track")