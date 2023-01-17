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
# calculate Euclidean distance 欧几里得距离

# you can use this function to calculate distance or use function cdist

def euclDistance(vector1, vector2):

    return np.sqrt(np.sum(np.power(vector2 - vector1, 2)))



# show your cluster (only available with 2-D data) 画出集群

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

def my_k_means(dataSet, k):   #dataSet:坐标，80对数

    centroids = list()        #记录质心坐标的表，4对数

    clusterAssment = list()   #记录聚类结果的表，80个数，0-3

    for i in range(len(dataSet)):

        clusterAssment.append([np.random.randint(k)]) #产生一个[0，k)随机整数

    for i in range(k):

        centroids.append(np.random.random(2))         #产生两个[0,1)随机数

    

    # TO DO

    i=0                      #迭代次数

    iteration=0              #进化次数 

    print=5                  #打印代数

    NSample=dataSet.shape[0] #样本个数

    distance=np.zeros(k)     #距离矩阵

    flag=True                #循环判断

    clusterMin=clusterAssment[:]#浅拷贝



    while flag:

        i+=1

        flag=False #小于五代或者未达最佳则继续循环

        #计算最佳距质心距离

        for j in range(NSample):

            for m in range(0,k):

                distance[m]=euclDistance(centroids[m],dataSet[j])

            clusterAssment[j]=[np.argmin(distance)]  

        #更新质心位置

        if clusterMin!=clusterAssment:

            iteration+=1

            clusterMin=clusterAssment[:]

            flag=True

            for m in range(0,k):

                x=0

                y=0

                n=0

                for j in range(NSample):

                    if(clusterAssment[j][0]==m):

                        n+=1

                        x+=dataSet[j][0,0]

                        y+=dataSet[j][0,1]

                if n!=0:

                    centroids[m][0]=x/n

                    centroids[m][1]=y/n    

        #打印前五代

        if i<=print:

            flag=True

            showCluster(dataSet, k, np.mat(centroids), np.mat(clusterAssment))            

    return np.mat(centroids), np.mat(clusterAssment),iteration



k = 4

centroids, clusterAssment,iteration= my_k_means(dataSet, k)

print('show the result of my_k_means ( needed iteration=',iteration,')')

showCluster(dataSet, k, centroids, clusterAssment)