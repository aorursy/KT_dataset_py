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





# calculate distance between each datapoint / each centroid to get new centroid    

def calnewcentroid(centroids, dataset, k):

    cluster = []

    for point in dataset:

        eucld = []  # euclid distance

        for i in range(k):

            eucld.append(euclDistance(centroids[i], point))

        cluster.append(np.argmin(eucld)) # smallest distance

    

    cluster = np.reshape(np.matrix(cluster),[-1, 1])   # change format

    newcentroid = []

    distance = 0

    

    # for each cluster, cal new centroid

    for i in range(k):  

        num = 0

        nx = 0

        ny = 0

        for j in range(len(cluster)):

            if cluster[j][0] == i:

                nx += dataset[j,0]   # x-axis and y-axis

                ny += dataset[j,1]

                num += 1

        if num == 0:

            newcentroid.append(centroids[i])

        else:

            newcentroid.append([nx/num, ny/num])

        distance += euclDistance(centroids[i], np.matrix([nx/num, ny/num]))  # total distance between now and previous

        

    newcentroid = np.matrix(newcentroid)

    return newcentroid, distance ,cluster





def my_k_means(dataSet, k):

    centroids = list()  # k centroids

    clusterAssment = list()  # cluster a point belongs to

    

    for i in range(len(dataSet)):

        clusterAssment.append([np.random.randint(k)])

    for i in range(k):

        centroids.append(np.random.random(2))

    

    distance = 1

    iters = 0

    while distance != 0: # if centroids no change, stop 

        centroids ,distance, clusterAssment = calnewcentroid(centroids, dataSet, k)

        if iters < 5:

            print('Iteration:',iters)

            showCluster(dataSet, k, centroids, clusterAssment)

            iters += 1



    

    return np.mat(centroids), np.mat(clusterAssment)



k = 4

centroids, clusterAssment = my_k_means(dataSet, k)

print('show the result of my_k_means ...')

showCluster(dataSet, k, centroids, clusterAssment)