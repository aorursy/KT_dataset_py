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

# 构建聚簇中心，取k个(此例中为4)随机质心



from numpy import *



def my_k_means(dataSet, k):

    centroids = list() #元素是array

    clusterAssment = list()

    for i in range(len(dataSet)):

        clusterAssment.append([np.random.randint(k)])

    for i in range(k):

        centroids.append(np.random.random(2))

    m,n = np.shape(dataSet)

    clusterAssment=np.mat(clusterAssment)

    

    

    for a in range(5):

        for i in range(m):  # 把每一个数据点划分到离它最近的中心点

            minDist = np.inf; minIndex = -1;

            for j in range(k):

                distJI = euclDistance(centroids[j], dataSet[i,:])

               

                if distJI < minDist:

                    minDist = distJI; minIndex = j  # 如果第i个数据点到第j个中心点更近，则将i归属为j

          

            clusterAssment[i] = [minIndex]  # 并将第i个数据点的分配情况存入字典

            

        suma=0

        sumb=0

        count=0

        for cent in range(k):   # 重新计算中心点

            for i in range (m):

                if (clusterAssment[i,0] ==cent):

                    suma= dataSet[i,:][0,0]+suma

                    sumb= dataSet[i,:][0,1]+sumb

                    count=count+1

            if(count!=0):

                suma=suma/count

                sumb=sumb/count

                centroids[cent]=[suma,sumb]

                suma=0

                sumb=0

                count=0

   # print(clusterAssment)

    return np.mat(centroids),np.mat(clusterAssment)

 



k = 4



centroids, clusterAssment = my_k_means(dataSet, k)

print('show the result of my_k_means ...')

showCluster(dataSet, k, centroids, clusterAssment)