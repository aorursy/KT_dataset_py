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
def cluster_assignment(dataSet, centroids):
    assignment_result = []
    #k = len(centroids)
    for point in dataSet:
        distance = []
        for center in centroids:
            distance.append(euclDistance(point, center))
        center_index = np.argmin(distance)
        assignment_result.append(center_index)
    #return np.transpose(np.matrix(assignment_result))
    return assignment_result

def get_centers(dataSet, assignment_result, k):
    centroids = []
    for i in range(k):
        points = [dataSet[j] for j,x in enumerate(assignment_result) if x==i]
        center = np.mean(points, 0).squeeze().tolist()
        centroids.append(center)
    return centroids
        
        
def my_k_means(dataSet, k):
    centroids_old = list()
    centroids_new = list()
    clusterAssment = list()
    epoch = 0
    #for i in range(len(dataSet)):
    #    clusterAssment.append([np.random.randint(k)])
    for i in range(k):
        centroids_new.append((np.random.random(2)*4-2).tolist())
    while centroids_old!=centroids_new:
        clusterAssment = cluster_assignment(dataSet, centroids_new)
        
        # draw 
        if epoch<5:
            showCluster(dataSet, k, np.mat(centroids_new), np.transpose(np.mat(clusterAssment)))
            
        centroids_old = centroids_new
        centroids_new = get_centers(dataSet, clusterAssment, k)
        epoch += 1
    
    # TO DO
    print('Total Epoch: %d'%epoch)
    return np.mat(centroids_new), np.transpose(np.mat(clusterAssment))

k = 4
print('show the result of my_k_means ...')
centroids, clusterAssment = my_k_means(dataSet, k)

#showCluster(dataSet, k, centroids, clusterAssment)
