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

print(dataSet.shape)

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

def my_k_means(dataSet, k):

    centroids = list()

    dis = np.zeros([len(dataSet),k+1])

    data = np.array(dataSet).squeeze()

    centroids_new = np.zeros([k,2])

    for i in range(k):

        centroids.append(np.random.random(2))

    centroids = np.array(centroids).squeeze()

    while True:

        for i in range(len(dataSet)):

            dis[i,0] = np.sqrt((data[i,0] - centroids[0,0])**2 + (data[i,1] - centroids[0,1])**2)

            dis[i,1] = np.sqrt((data[i,0] - centroids[1,0])**2 + (data[i,1] - centroids[1,1])**2)

            dis[i,2] = np.sqrt((data[i,0] - centroids[2,0])**2 + (data[i,1] - centroids[2,1])**2)

            dis[i,3] = np.sqrt((data[i,0] - centroids[3,0])**2 + (data[i,1] - centroids[3,1])**2)

            dis[i,4] = np.argmin(dis[i,:4])

        index1 = dis[:,4] == 0

        index2 = dis[:,4] == 1

        index3 = dis[:,4] == 2

        index4 = dis[:,4] == 3

        if any(index1):

            centroids_new[0,:] = [data[index1,0].mean(),data[index1,1].mean()] 

        else:

            centroids_new[0,:] = centroids[0,:]

        if any(index2):

            centroids_new[1,:] = [data[index2,0].mean(),data[index2,1].mean()] 

        else:

            centroids_new[1,:] = centroids[1,:]

        if any(index3):

            centroids_new[2,:] = [data[index3,0].mean(),data[index3,1].mean()] 

        else:

            centroids_new[2,:] = centroids[2,:]

        if any(index4):

            centroids_new[3,:] = [data[index4,0].mean(),data[index4,1].mean()] 

        else:

            centroids_new[3,:] = centroids[3,:]

        if((centroids == centroids_new).all()):

            break

        centroids = centroids_new.copy()

        clusterAssment = np.reshape(dis[:,4],[-1,1])

        showCluster(dataSet, k, centroids, clusterAssment)

    clusterAssment = np.reshape(dis[:,4],[-1,1])

    return centroids, clusterAssment



k = 4

centroids, clusterAssment = my_k_means(dataSet, k)

print('show the result of my_k_means ...')

showCluster(dataSet, k, centroids, clusterAssment)