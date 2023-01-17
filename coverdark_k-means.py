import numpy as np

from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
# calculate Euclidean distance

def euclDistance(vector1, vector2):

    return np.sqrt(np.sum(np.power(vector2 - vector1, 2)))
# init centroids with random samples  

def initCentroids(dataSet, k):

    numSamples, dim = dataSet.shape

    index = np.random.uniform(0, numSamples, k).astype(int)

    centroids = dataSet[index]

    return centroids
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
# k-means cluster

def kmeans(dataSet, k):  

    numSamples = dataSet.shape[0]

    # store which cluster this sample belongs to

    clusterAssment = np.zeros([numSamples, 1])

    clusterChanged = True



    ## step 1: init centroids

    centroids = initCentroids(dataSet, k)



    epoch = 0

    while clusterChanged:

        clusterChanged = False

        ## for each sample

        for i in range(numSamples):

            minDist  = float('inf')

            minIndex = 0

            # for each centroid

            # step 2: find the centroid who is closest  

            for j in range(k):  

                distance = euclDistance(centroids[j, :], dataSet[i, :])  

                if distance < minDist:  

                    minDist  = distance  

                    minIndex = j  

              

            ## step 3: update its cluster 

            if clusterAssment[i, 0] != minIndex:

                clusterChanged = True

                clusterAssment[i, :] = minIndex



        ## step 4: update centroids

        for j in range(k):

            pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0] == j)[0], :]

            centroids[j, :] = np.mean(pointsInCluster, axis=0)

        

        if epoch < 5:

            print('epoch: ' + str(epoch))

            showCluster(dataSet, k, centroids, clusterAssment)

        epoch = epoch + 1

        

    print ('Congratulations, cluster complete!')

    return centroids, clusterAssment
# k-means cluster

def kmeans_simple(dataSet, k):

    numSamples = dataSet.shape[0]

    clusterChanged = True

    clusterAssment = np.zeros([numSamples, 1])

    

    ## step 1: init centroids

    centroids = initCentroids(dataSet, k)



    while clusterChanged:

        clusterChanged = False

        # calculate pairwise distance

        distance = cdist(dataSet, centroids)



        # find the closest centroid for each sample

        tmpIndex = np.reshape(np.argmin(distance, 1), [-1, 1])

        

        # if any index changes, continue

        if (tmpIndex != clusterAssment).any():

            clusterChanged = True



        # update clusterAssment

        clusterAssment = tmpIndex



        # update centroids

        for j in range(k):

            pointsInCluster = dataSet[np.nonzero(clusterAssment == j)[0], :]

            centroids[j, :] = np.mean(pointsInCluster, 0)



    print ('Congratulations, cluster complete!')  

    return centroids, clusterAssment
def customReadFile(fileName):

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
## step 1: load data

fileIn = '../input/testSet.txt'

print ('Step 1: Load data ' + fileIn + '...')

dataSet = customReadFile(fileIn)

print('Number of samples: ' + str(dataSet.shape[0]))



## step 2: clustering...  

print ("Step 2: clustering..."  )

k = 4

centroids, clusterAssment = kmeans(dataSet, k)

# centroids, clusterAssment = kmeans_simple(dataSet, k)

# clusteringResult = KMeans(n_clusters=k).fit(dataSet)

# clusterAssment = np.reshape(clusteringResult.labels_, [-1, 1])

# centroids = clusteringResult.cluster_centers_



## step 3: show the result

print ("Step 3: show the result..."  )

showCluster(dataSet, k, centroids, clusterAssment)