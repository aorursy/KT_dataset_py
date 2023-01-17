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


def init(dataSet, k):
    num, dim = dataSet.shape
    init_centroids = np.zeros((k, dim))
    for i in range(k):
        rand = int(np.random.uniform(0, num))
        init_centroids[i,:] = dataSet[rand,:]
    return init_centroids, num

def calculate_dis(a, b):
    return np.sqrt(np.sum(np.power(a-b, 2)))

def my_k_means(dataSet, k):
    centroids, num = init(dataSet, k)
    clusterAssment = np.zeros((num, 2))
    flag = True

    while flag:
        flag = False
        for i in range(num):
            min_Distance = float('inf')
            min_Index = 0

            for m in range(k):
                dis = calculate_dis(centroids[m,:], dataSet[i,:])
                if (dis< min_Distance):
                    min_Distance = dis
                    min_Index = m

            if (clusterAssment[i,0] != min_Index):
                flag = True
                clusterAssment[i,:] = min_Index, pow(min_Distance, 2)

        for m in range(k):
            binary_value = (clusterAssment[:,0] == m)
            nonzero_index = np.nonzero(binary_value)[0]
            new_centroid = dataSet[nonzero_index,:]
            centroids[m,:] = np.mean(new_centroid, axis=0)


        showCluster(dataSet, k, centroids, clusterAssment)
    return centroids, clusterAssment


k = 4
centroids, clusterAssment = my_k_means(dataSet, k)
