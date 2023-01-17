import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
dfTrain = pd.read_csv('../input/UnclusteredData.csv')   #Training Dataset

dfTrain.head()
pd.plotting.scatter_matrix(dfTrain, alpha=1, diagonal='hist',color='k')

plt.show()
def extractFeatures(df):

    df_Features=df.iloc[:,0:2]

    X=df_Features.values

    Y=np.zeros((len(df_Features),1))       #All are assigned to K= 1 cluster

    return X,Y
X,Y=extractFeatures(dfTrain)
cmap = ListedColormap(['black','magenta', 'red','green','orange']) 

plt.scatter(dfTrain.loc[:,['X1']].values,dfTrain.loc[:,['X2']].values, c=Y,cmap=cmap,marker="o")

plt.show()
def initCentroids(X,K):

    rand_indices = np.random.permutation(X.shape[0])

    init_centroids = X[rand_indices[:K],:]

    return init_centroids
def assignCentroids(X,centroids):    

    K=centroids.shape[0]

    #Assign index to each training set

    Y=np.zeros((X.shape[0],1))

    for i in range(len(X[:,0:1])): 

        Prev_Distance=np.linalg.norm( X[i,:]-centroids[0,:])

        for j in range(1,K):

            Current_Distance=np.linalg.norm( X[i,:]-centroids[j,:])

            if(Current_Distance<=Prev_Distance):

                Y[i]=j

                Prev_Distance=Current_Distance

    return Y
def updateNewCentroids(X,Y,K):    

    centroids = np.zeros((K,X.shape[1]))

    for j in range(0,K):

        sumC=np.zeros((1,X.shape[1]))

        countC=0

        for i in range(len(X[:,0:1])): 

            if (Y[i]==j):

                sumC=sumC+X[i,:]

                countC=countC+1

        if (countC!=0):

            centroids[j,:]=(1/countC)*sumC



    return centroids
K=5

init_centroids = initCentroids(X,K)



centroids=init_centroids    #start with Initial Centroid

centroidsHistory=[init_centroids]

while True:

    Y = assignCentroids(X,centroids)      #Assignment Step

    newCentroids =updateNewCentroids(X,Y,K)   #Update New Centroids Step

    if ((newCentroids==centroids).all()):

        break

    else:

        centroids=newCentroids

        centroidsHistory.append(newCentroids)



plt.scatter(X[:,0:1],X[:,1:2],c=Y,cmap=cmap,marker=".")

#plot Centroid History with line how centorids are moving

for i in range(K):

    histCentroids=np.array( centroidsHistory)[:,i,:]

    plt.plot(histCentroids[:,0:1],histCentroids[:,1:2],color='b',linestyle='dashed')

    

#Final Centroid

plt.scatter(centroids[:,0:1],centroids[:,1:2],color='b',marker="o",edgecolor='b')

plt.title('K-mean (K='+str(K)+')')

plt.show()

maxK=10

meanvalues=np.zeros((maxK))

for K in (range(1,maxK)):

    rand_indices = np.random.permutation(X.shape[0])

    init_centroids = X[rand_indices[:K],:]

    centroids=init_centroids    #start with Initial Centroid

    while True:

        Y = assignCentroids(X,centroids)      #Assignment Step

        newCentroids =updateNewCentroids(X,Y,K)   #Update New Centroids Step

        if ((newCentroids==centroids).all()):

            break

        else:

            centroids=newCentroids

    ##############

    meanvalues[K]=0



    for i in range(K):

        distanceValues=np.linalg.norm(X[np.where(Y==i)[0]]-centroids[i,:],axis=1)

        meanvalues[K]=meanvalues[K]+np.mean(distanceValues)/K

        

#######################        

bestK=0

distanceOfKfromOrigin=0

for K in (range(1,maxK)):

    NewdistanceOfKfromOrigin=np.linalg.norm(np.array([K,meanvalues[K]]))

    #print(K,NewdistanceOfKfromOrigin)

    if NewdistanceOfKfromOrigin<=distanceOfKfromOrigin or distanceOfKfromOrigin==0:

        distanceOfKfromOrigin=NewdistanceOfKfromOrigin

        bestK=K

bestK=bestK+1 #next K

###########################

plt.plot(range(1,maxK),meanvalues[1:maxK],color='b')    

plt.scatter(bestK,meanvalues[bestK],color='r',marker="X")     

plt.ylabel('Avg Distance from Centroids')

plt.xlabel('Values of K')

plt.title("Elbow Method where best K ="+ str(bestK))

plt.show()