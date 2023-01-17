import numpy as np

import matplotlib.pyplot as plt

from PIL import Image
def findClosestCentroids(X, centroids, K):

# Given a data set X with m samples each having n features, and

# a set of K centroid points with n features each, this function

# computes the index of the centroid which is closest to each sample and

# stores it in a index array called idx.

    

    (m, n) = np.shape(X)

    idx  = np.zeros(m, dtype=np.int16)

    dist = np.zeros(K, dtype=np.double)

    for i in range(0, m):

        for j in range(0, K):

            dist[j] = 0

            for k in range(0, n):

                dist[j] += (X[i][k] - centroids[j][k]) ** 2

        idx[i] = np.argmin(dist)

    return idx
def computeCentroids(X, idx, K):

# Given a data set X with m samples and n features, and an index

# array idx which contains the grouping of all samples from a range

# 1 to K, derived by proximity to initial centroid points, this function

# computes a new set of centroids, based on the grouping idx.

    

    (m, n) = np.shape(X)

    centroids = np.zeros((K, n), dtype=np.double)

    for i in range(0, m):

        for j in range(0, n):

            centroids[idx[i], j] += X[i][j]

    for i in range(0, K):

        sum = 0

        for k in range(0, m):

            if idx[k] == i:

                sum += 1

        for j in range(0, n):

            centroids[i][j] /= sum

    return centroids
def KMeansInitCentroid(X, K):

# This function basically initializes K centroids by randomly

# picking K samples from the data set and associating it with 

# initial centroid positions.



    (m, n) = np.shape(X)

    centroids = np.zeros((K, n), dtype=np.double)

    randperm = np.random.permutation(m)

    for i in range(0, K):

        for j in range(0, n):

            centroids[i][j] = X[randperm[i]][j]

    return centroids
def runKMeans(X, initial_centroids, K, max_iters):

# This is the heart of the Algorithm, the K Means algorithm implementation.

# It takes as input the data set, and a set of K randomly initialized centroids

# and by repeatatively iterating through the findClosestCentroids and ComputeCentroids

# subroutine it finally effective clusters the entire data set into separate clusters of subsets.



    m = np.shape(X)

    centroids = initial_centroids

    idx = np.zeros(m, dtype=np.int16)

    for i in range(1, max_iters+1):

        print(i, max_iters)

        idx = findClosestCentroids(X, centroids, K)

        centroids = computeCentroids(X, idx, K)

    return centroids, idx
# In this code block, we import the classical lena image in rgb format with a pixel

# resolution of 512 * 512 * 3(R,G,B takes values between 0 to 255 - 24bytes storage),

# After reshaping the array, we cluster the entire data set into 16 subgroups using 

# 10 iterations of the K-means algorithm. We are left with an index array of 

# 512 * 512 * 1 (takes values from 1 to 16 - 4 bytes storage) plus a centroid array (which

# stores the new colors) and has a size of 16 * 3(R,G,B takes values between 0 to 255 - 24bytes storage)

# Effective compression is 6 times !



fname = "/kaggle/input/lenargb128/lena128.jpg"

im = Image.open(fname).convert('RGB')

lena = np.asarray(im)

(L, M, N) = np.shape(lena)

X = np.reshape(lena, (L*M, N), order='F')

X = np.true_divide(X, 255)

K = 16

max_iter = 10



initial_centroids = KMeansInitCentroid(X, K)

centroids, idx = runKMeans(X, initial_centroids, K, max_iter)

idx = findClosestCentroids(X, centroids, K)
X_rebuild = np.zeros((L*M, N), dtype=np.double)

for i in range(0, L*M):

    X_rebuild[i] = centroids[idx[i]]

X_rebuild = np.reshape(X_rebuild, (L, M, N), order='F')

X = np.reshape(X, (L, M, N), order='F')



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,20))

ax1.imshow(X)

ax1.axis('off')

ax1.title.set_text('LENA RGB')

ax2.imshow(X_rebuild)

ax2.axis('off')

ax2.title.set_text('LENA Rebuilt with 16 Mean Colors')

plt.show()
