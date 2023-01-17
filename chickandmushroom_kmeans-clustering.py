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



#print(dataSet)
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

#print('clu_res',clusteringResult)

clusterAssment = np.reshape(clusteringResult.labels_, [-1, 1])

#print('clu_ass',clusterAssment)

centroids = clusteringResult.cluster_centers_

#print('cent',centroids)



print('show the result of sklearn ...')

showCluster(dataSet, k, centroids, clusterAssment)
# 1. finish function my_k_means;

# 2. Plot the results of the first five iterations (use function showCluster)

def my_k_means(dataSet, k, n_iter):

    centroids = list()

    clusterAssment = list()

    dist = list()

    

    for i in range(len(dataSet)):

        clusterAssment.append([np.random.randint(k)])

    for i in range(k):

        centroids.append(np.random.random(2))

    

    print('The Initialization of K-means Clustering:')

    showCluster(dataSet, k, np.mat(centroids),np.mat(clusterAssment))

    

    for n in range(n_iter):

        # Assignment for Each Points

        for i in range(len(dataSet)):

            vec1 = np.array(dataSet[i])

            dist = list()

            for j in range(k):

                dist.append(euclDistance(vec1,centroids[j]))

            #print(dist)

            clusterAssment[i] = [np.argmin(np.array(dist))]

        #print(clusterAssment)

    

        # Update the Position of Centroids

        

        for j in range(k):

            belongs_x = list()

            belongs_y = list()

            for i in range(len(dataSet)):

                if clusterAssment[i]==[j]:

                    belongs_x.append(dataSet[i,0])

                    belongs_y.append(dataSet[i,1])

            

            cent_x = np.mean(np.array(belongs_x))

            cent_y = np.mean(np.array(belongs_y))

            centroids[j] = np.array([cent_x,cent_y])

        #print(centroids)

        

        print('Clustering Result after Iteration ',n)

        showCluster(dataSet, k, np.mat(centroids),np.mat(clusterAssment))

        

    return np.mat(centroids), np.mat(clusterAssment)



k = 4

n_iter = 5

centroids, clusterAssment = my_k_means(dataSet, k, n_iter)

print('Show the result after 5 iterations of clustering in my_k_means:')

showCluster(dataSet, k, centroids, clusterAssment)