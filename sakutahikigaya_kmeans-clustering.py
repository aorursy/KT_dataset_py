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

    flag1 = flag2 = 1

    # TO DO

    num = 0

    while(flag1 == 1 | flag2 == 1):

        #E-step



        num += 1

        if num < 6:

            showCluster(dataSet, k, np.mat(centroids), np.mat(clusterAssment))

        dis = list()

        for i in range(len(dataSet)):

            dis.append(list())

            for j in range(k):

                dis[i].append(np.sqrt(np.sum(np.square(np.array(dataSet[i]) - np.array(centroids[j])))))

        cls_point = list()

        for i in range(len(dataSet)):

            cls_point.append([dis[i].index(min(dis[i]))])

        if cls_point == clusterAssment:

            flag1 = 0

        else:

            clusterAssment = cls_point

            flag1 = 1





        #M-step

        #find points in different clusters

        cls_cls =list()

        for i in range(k):

            cls_cls.append([j for j in range(len(cls_point)) if cls_point[j] == [i]])

        #calculate center of k-clusters

        centroids_tmp = list()

        for i in range(k):

            sum = [0,0]

            if len(cls_cls[i])> 0:

                for j in range(len(cls_cls[i])):

                    sum = sum + dataSet[(cls_cls[i][j])]

                [sum_tmp] = np.array(sum/len(cls_cls[i]))

                centroids_tmp.append(sum_tmp)

            else:

                centroids_tmp.append(centroids[i])

        flag2_1 =0

        for i in range(len(centroids_tmp)):

            if centroids_tmp[i][0] == centroids[i][0]:

                flag2_1+=1

        if flag2_1 == len(centroids_tmp):

            flag2 = 0

        else:

            flag2 = 1

            centroids = centroids_tmp

    return np.mat(centroids), np.mat(clusterAssment)





k = 4



print('show the result of my_k_means ...')

iterations = 5

centroids, clusterAssment = my_k_means(dataSet, k)

print(centroids)

print(clusterAssment)

showCluster(dataSet, k, centroids, clusterAssment)


