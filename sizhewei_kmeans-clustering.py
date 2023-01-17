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

        # https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.randint.html

        clusterAssment.append([np.random.randint(k)])

    for i in range(k):

        # https://numpy.org/doc/1.18/reference/random/generated/numpy.random.random.html

        centroids.append(np.random.random(2))



    # TO DO

    num_dataSet = len(dataSet)



    def _converged(cent1, cent2):

        len1 = len(cent1)

        len2 = len(cent2)

        if (len1 != len2):

            return False

        for i in range(len1):

            if (not (cent1[i] == cent2[i]).all()):

                return False



        return True



    iritates = 0

    while (True):

        if (iritates % 100 == 0):

            print(iritates)

        newcentroids = centroids[:]



        for i in range(num_dataSet):

            tmp_cluster = -1

            tmp_distance = float("inf")

            for j in range(k):

                if (euclDistance(centroids[j], dataSet[i]) < tmp_distance):

                    tmp_cluster = j

                    tmp_distance = euclDistance(centroids[j], dataSet[i])

            clusterAssment[i] = tmp_cluster

        

        if(iritates<5):

            print('show the result of my_k_means at {} iteration:'.format(iritates+1))

            showCluster(dataSet, k, np.mat(centroids), np.mat(clusterAssment).T)

        

        for i in range(k):

            tmp_sum = np.zeros((1, dataSet.shape[1]))

            for j in range(num_dataSet):

                if (clusterAssment[j] == i):

                    tmp_sum += dataSet[j]

            if (clusterAssment.count(i)==0):

                newcent = tmp_sum

            else:

                newcent = tmp_sum / clusterAssment.count(i)

            newcentroids[i] = np.squeeze(newcent)



        if (_converged(newcentroids, centroids)):

#             print(newcentroids)

#             print(centroids)

            break



        centroids = newcentroids

        iritates += 1



    return np.mat(centroids), np.mat(clusterAssment).T



k = 4

centroids, clusterAssment = my_k_means(dataSet, k)

print('show the result of my_k_means ...')

showCluster(dataSet, k, centroids, clusterAssment)