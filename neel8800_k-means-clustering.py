# Import Some Library

from copy import deepcopy

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt
# Randomly Initialize Four Centers, the Model Should Predict Similar Results

center_1 = np.array([1,1])

center_2 = np.array([5,5])

center_3 = np.array([8,1])

center_4 = np.array([11,5])





# Generate Random Data

data_1 = np.random.randn(200, 2) + center_1

data_2 = np.random.randn(200,2) + center_2

data_3 = np.random.randn(200,2) + center_3

data_4 = np.random.randn(200,2) + center_4





# Final Data

data = np.concatenate((data_1, data_2, data_3, data_4), axis = 0)



# Visualization

plt.scatter(data[:,0], data[:,1], s=8)
# Number of Clusters k

k = 4



# Number of Training Data

n = data.shape[0]



# Number of Features in the Data

c = data.shape[1]



# Generate Random Centers

mean = np.mean(data, axis = 0)

std = np.std(data, axis = 0)

centers = np.random.randn(k,c)*std + mean



# Viasualization Data along with Centers

plt.scatter(data[:,0], data[:,1], s=7)

plt.scatter(centers[:,0], centers[:,1], marker='*', c='red', s=150)
# Main K_Means_Clustering Function



# Initialize Center_old and Center_new to Store Center

centers_old = np.zeros(centers.shape)

centers_new = deepcopy(centers)



# Initialize Cluster and Distance

clusters = np.zeros(n)

distances = np.zeros((n,k))





# Initialize Error

error = np.linalg.norm(centers_new - centers_old)



# Compute Until error Become Zero

while error != 0:

    

    # Measure the Distance From Every Center to Every Data

    for i in range(k):

        distances[:,i] = np.linalg.norm(data - centers[i], axis=1)

    

    # Assign All Data to Closest Center

    clusters = np.argmin(distances, axis = 1)

    

    centers_old = deepcopy(centers_new)

    

    # Calculate Mean for Every Cluster and Update the Center

    for i in range(k):

        centers_new[i] = np.mean(data[clusters == i], axis=0)

    

    # New Error

    error = np.linalg.norm(centers_new - centers_old)

    

centers_new
# Plot the data and the centers generated as random

plt.scatter(data[:,0], data[:,1], s=7)

plt.scatter(centers_new[:,0], centers_new[:,1], marker='*', c='red', s=150)
# Separate Data According to Nearest Center

data1 = data[clusters==0]

data2 = data[clusters==1]

data3 = data[clusters==2]

data4 = data[clusters==3]



# Visualization

plt.scatter(data1[:,0], data1[:,1], s=8, color='b')

plt.scatter(data2[:,0], data2[:,1], s=8, color='g')

plt.scatter(data3[:,0], data3[:,1], s=8, color='orange')

plt.scatter(data4[:,0], data4[:,1], s=8, color='yellow')



plt.scatter(centers_new[:,0], centers_new[:, 1], color='red', s=125, marker='*')