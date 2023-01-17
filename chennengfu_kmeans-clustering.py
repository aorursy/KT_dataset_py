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

    ##init parameters

    samples = dataSet.shape[0]

    isChanged = True

    epoch = 0

    Regenerate_flag = 0

    

    #do k-means clustering

    while isChanged:

        epoch += 1

        isChanged = False

        

        ##centroids degenerate to 3 or less, which means random initialization of centroids failed

        ##regenerate initialized centroids

        if Regenerate_flag == 1:

            centroids = list()

            for i in range(k):

                centroids.append(np.random.random(2))

                Regenerate_flag = 0

                

        ##Reassign cluster centroid to every samples 

        for i in range(samples):

            mindist = float('inf')

            minpos = 0

            for j in range(k):

                Lk = euclDistance(dataSet[i,:], centroids[j])

                if Lk < mindist:

                    mindist = Lk

                    minpos = j

            

            ##judge whether a sample changes its centroids

            if clusterAssment[i] != [minpos]:

                isChanged = True

                clusterAssment[i] = [minpos]

        

        ##Plot the results of the first five iterations

        if epoch <= 5:

            print('epoch '+str(epoch))

            showCluster(dataSet, k, np.mat(centroids), np.mat(clusterAssment))

            

        ##refresh the centroids

        for j in range(k):

            idx = []

            for index,value in enumerate(clusterAssment):

                if value == [j]:

                    idx.append(index)

                    

            ##judge whether centroids decrease 

            if idx == []:

                Regenerate_flag = 1

                break

                

            M = np.mean(dataSet[idx,:], axis = 0)

            centroids[j] = np.squeeze(np.asarray(M))

        

    return np.mat(centroids), np.mat(clusterAssment)



k = 4

centroids, clusterAssment = my_k_means(dataSet, k)

print('show the result of my_k_means ...')

showCluster(dataSet, k, centroids, clusterAssment)