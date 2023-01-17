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
print(np.array(dataSet)[0])

print()

#print(dataSet)
# 1. finish function my_k_means;

# 2. Plot the results of the first five iterations (use function showCluster)

def my_k_means(dataSet, k):

    centroids = list()

    clusterAssment = list()

    for i in range(len(dataSet)):

        clusterAssment.append([np.random.randint(k)])  #代表各个点属于哪一聚类，二维列表，值为0-3，表长80

    for i in range(k):

        centroids.append(np.random.random(2))#一维质心位置矩阵，表长为4

    #print(clusterAssment[0][0])

    

    # TO DO

    #迭代五次

    for i in range(5):

        #计算距离

        for j in range(len(dataSet)):

            mindist=[]

            for n in range(k):

                mindist.append(euclDistance(dataSet[j],centroids[n]))

            clusterAssment[j]=[np.argmin(mindist)]

        #更新质心

        dataSet_array=np.array(dataSet)

        for n in range(k):

            x_num=0

            y_num=0

            x_total=0

            y_total=0

            for j in range(len(dataSet)):

                if(clusterAssment[j][0]==n):

                    x_num+=1

                    y_num+=1

                    x_total+=dataSet_array[j][0]

                    y_total+=dataSet_array[j][1]

            #符合更新条件则更新

            if x_num!=0 and y_num!=0:

                centroids[n][0]=x_total/x_num

                centroids[n][1]=y_total/y_num

        #显示结果

        showCluster(dataSet, k, np.mat(centroids), np.mat(clusterAssment))

    return np.mat(centroids), np.mat(clusterAssment)#返回矩阵



k = 4

print('show the first five result of my_k_means ...')

centroids, clusterAssment = my_k_means(dataSet, k)



#showCluster(dataSet, k, centroids, clusterAssment)