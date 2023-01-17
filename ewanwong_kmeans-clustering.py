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
# 1. finish function my_k_means;

# 2. Plot the results of the first five iterations (use function showCluster)

import random

def assign(X, centroids):

    k = len(centroids)

    distance_list = []

    for i in range(k):

        distance_list.append(euclDistance(X, centroids[i]))

    return distance_list.index(min(distance_list))



def my_k_means(dataSet, k, iters):

    centroids = list()

    clusterAssment = list()

    m = len(dataSet)

    # initialize clusterAssment and centroids

    for i in range(len(dataSet)):

        clusterAssment.append([np.random.randint(k)])

    random_list = random.sample(range(len(dataSet)), k)

    for i in range(k):

        centroids.append(dataSet[random_list[i], :])

    centroids = np.array(centroids)

    

    for i in range(iters):

        # assign dataSet

        for j in range(m):

            clusterAssment[j] = [assign(dataSet[j], centroids)]

        # calculate new centroids

        dis_list = [0 for _ in range(k)]

        sum_list = [[0, 0] for _ in range(k)]

        for j in range(m):

            assignment = clusterAssment[j][0]

            dis_list[assignment] += 1

            sum_list[assignment][0] += dataSet[j, 0]

            sum_list[assignment][1] += dataSet[j, 1]

        sum_list = np.array(sum_list)

        for j in range(k):

            centroids[j] = [sum_list[j][0] / dis_list[j], sum_list[j][1] / dis_list[j]]

        if i in range(5):

            showCluster(dataSet, k, np.mat(centroids), np.mat(clusterAssment))



    return np.mat(centroids), np.mat(clusterAssment)



k = 4

centroids, clusterAssment = my_k_means(dataSet, k, 100)

print('show the final result of my_k_means ...')

showCluster(dataSet, k, centroids, clusterAssment)