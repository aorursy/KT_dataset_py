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
def my_k_means(dataSet, k):
    centroids = list()
    clusterAssment = list()
    for i in range(len(dataSet)):
        clusterAssment.append([np.random.randint(k)])
    for i in range(k):
        centroids.append(np.random.random(2))
    # TO DO
    iteration = 1
    while(True):
        # Classification
        for i in range(len(dataSet)):
            min_dist = euclDistance(dataSet[i], centroids[0])
            index = 0
            for j in range(1,k):
                Dist = euclDistance(dataSet[i], centroids[j])
                if Dist < min_dist:
                    min_dist = Dist
                    index = j
            clusterAssment[i][0] = index
            
        # update cluster center
        end_flag = 0
        for i in range(k):
            x_mean = 0
            y_mean = 0
            sums = 0
            for j in range(len(dataSet)):
                if clusterAssment[j][0] == i:
                    x_mean += dataSet.item(j,0)
                    y_mean += dataSet.item(j,1)
                    sums += 1
            if (centroids[i][0] == x_mean/sums) & (centroids[i][1] == y_mean/sums):
                end_flag += 1
            else:
                centroids[i][0] = x_mean/sums
                centroids[i][1] = y_mean/sums

        # show the first five results
        if iteration <= 5:
            print("Iteration = %d" % iteration)
            showCluster(dataSet, k, np.mat(centroids), np.mat(clusterAssment))
        iteration += 1
        
        # If the cluster center is no longer updated, quit
        if end_flag == k:
            break

    return np.mat(centroids), np.mat(clusterAssment)

k = 4
centroids, clusterAssment = my_k_means(dataSet, k)
print('show the result of my_k_means ...')
showCluster(dataSet, k, centroids, clusterAssment)