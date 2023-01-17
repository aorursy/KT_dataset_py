import numpy as np

from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans



def ReadFile(fileName):

    fileIn = open(fileName, 'r')

    dataSet = []

    for line in fileIn.readlines():

        temp=[]

        lineArr = line.strip().split('\t')

        temp.append(float(lineArr[0]))

        temp.append(float(lineArr[1]))

        dataSet.append(temp)

    fileIn.close()

    return np.mat(dataSet)



dataSet = ReadFile('../input/data.txt')

plt.scatter(np.array(dataSet)[:, 0], np.array(dataSet)[:, 1])

plt.show()

print('Data set size:', len(dataSet))
# calculate Euclidean distance

# you can use this function to calculate distance or use function cdist

def euclDistance(vector1, vector2):

    return np.sqrt(np.sum(np.power(vector2 - vector1, 2)))



# show your cluster (only available with 2-D data) 

def showCluster(dataSet, k, centroids, clusterAssment):  

    numSamples, dim = dataSet.shape

    if dim != 2:

        print ("Sorry! I can not draw because the dimension of your data is not 2!")  

        return 1  

  

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']  

    if k > len(mark):

        print ("Sorry! Your k is too large!")  

        return 1



    # draw all samples

    for i in range(numSamples):

        # assign colors for samples

        markIndex = int(clusterAssment[i, 0])

        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])



    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']  

    # draw the centroids

    for i in range(k):

        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12)  



    plt.show()
k = 4

clusteringResult = KMeans(n_clusters=k).fit(dataSet)

clusterAssment = np.reshape(clusteringResult.labels_, [-1, 1])

centroids = clusteringResult.cluster_centers_



print('show the result of sklearn ...')

showCluster(dataSet, k, centroids, clusterAssment)

print(clusterAssment.shape)
# 1. finish function my_k_means;

# 2. Plot the results of the first five iterations (use function showCluster)

def my_k_means(dataSet, k):

    # centroids = list()

    # clusterAssment = list()

    # for i in range(len(dataSet)):

        # clusterAssment.append([np.random.randint(k)])

    # for i in range(k):

        # centroids.append(np.random.random(2))

    centroids = np.random.rand(k, 2)

    clusterAssment = np.random.randint(low=0, high=k, size=len(dataSet))

    distances = np.zeros((len(dataSet), k))

    MAX_ITER = 100

    it = 0

    # TO DO

    while it < MAX_ITER:

        prev_centroids = centroids.copy()

        for j in range(k):

            distances[:, j] = np.sqrt((np.power(centroids[j] - dataSet, 2)).sum(axis=1)).squeeze()

        clusterAssment = np.argmin(distances, axis=1)

        for j in range(k):

            if (clusterAssment == j).sum():

                centroids[j] = dataSet[clusterAssment == j].mean(axis=0).squeeze()

        clusterAssment = np.expand_dims(clusterAssment, 1)

        if it < 5:

            showCluster(dataSet, k, centroids, clusterAssment)

        it += 1

        if not (centroids != prev_centroids).sum():

            break

    

    return np.mat(centroids), np.mat(clusterAssment)



k = 4

centroids, clusterAssment = my_k_means(dataSet, k)

print('show the result of my_k_means ...')

showCluster(dataSet, k, centroids, clusterAssment)