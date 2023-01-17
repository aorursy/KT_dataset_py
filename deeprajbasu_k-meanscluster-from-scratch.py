# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

df = pd.read_csv("../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")



df.tail()
X = df.iloc[:, [3, 4]].values

X
import seaborn as sns

import matplotlib.pyplot as plt







sns.set(style='darkgrid')



fig, ax = plt.subplots(figsize=(20,7.75))

sns.scatterplot(x="Annual Income (k$)", y="Spending Score (1-100)", data=df,size="Age",sizes=(55,700))

X[5]
#variables for number of samples, and number of feature for each sample



m=X.shape[0] #number of training examples

n=X.shape[1] #number of features. Here n=2

n_iter=100
#number of clusters that we would like to divide the data into



K=5 # number of clusters
import random as rd

##array that will contain all the centroids



Centroids=np.array([]).reshape(n,0) 

Centroids



#initialize k number of centroids by randomly selecting datapoints

for i in range(K):

    rand=rd.randint(0,m-1)#generate random index positing between 0 and last index of data

    

    Centroids=np.c_[Centroids,X[rand]]# add this random data point to our centroid array

Centroids

#we can acess the points like so 

#if we want to retrieve the first centroid

Centroids
Centroids[:,0]
Output={}
EuclidianDistance=np.array([]).reshape(m,0) #m, again is the number of data samples



for k in range(K):# iterating for k number of times, k is the number of clusters we want 

       tempDist=np.sum((X-Centroids[:,k])**2,axis=1)#generating eucledian distance for all centroid  with current data pooit 

       EuclidianDistance=np.c_[EuclidianDistance,tempDist]#adding this list of distances to our matrix





####IMPORTANT####

C=np.argmin(EuclidianDistance,axis=1)+1 #Storing the index of the minimum distance in this data point

## therefore, for the datapoint X[i] the group it belongs to is C[i]



###we can then take the mean of this group and update the centroids
EuclidianDistance[:5] # ditance of each data point from the centroids



C[:15] #position of the closest cluster for each datapoint



#example, for the first data point, its closest to the third cluster, 

#based on this we shall re group the data
Y={}# temporary dictionary to hold our solution for this iteration

for k in range(K):

    Y[k+1]=np.array([]).reshape(2,0) # populate our dictionary with k elements(representing our clusters), each having a default key and a 2,0 array

    

####IMPORTANT######



for i in range(m):

     Y[C[i]]=np.c_[Y[C[i]],X[i]]# addding all the datapoints to chosen clusters in Y, based on their stored indexes in C

     

for k in range(K):

    Y[k+1]=Y[k+1].T #reshaping all the points to look like 2,0 arrays

    

for k in range(K):

     Centroids[:,k]=np.mean(Y[k+1],axis=0) # assiginng the mean of each group as new centroids
for i in range(n_iter):

    

    EuclidianDistance=np.array([]).reshape(m,0) #m, again is the number of data samples



    for k in range(K):# iterating for k number of times, k is the number of clusters we want 

       tempDist=np.sum((X-Centroids[:,k])**2,axis=1)#distance between current datapoint and each centroid

       EuclidianDistance=np.c_[EuclidianDistance,tempDist]#add distance to our list

    #the result is for each datapoint, the distances  to each centroid





    ####IMPORTANT####

    #!!!!!!!!!!!!!!!#

    

    C=np.argmin(EuclidianDistance,axis=1)+1 #storing the index of the minimum distance fron the clusters in C : the index position denotes which centroid, or cluster

    #this list contains the value of the nearest centroid for each datapoint, 

    #you can say it contains the info for which datapoint belongs to which cluster

    

    

    #the position here is which datapoint for X, and the Value is the cluster it belongs too   

    

    #***********#

    ## therefore, for the datapoint X[i] the group it belongs to is C[i]

    

    

    

    Y={}# temporary dictionary to hold our solution for this iteration

    for k in range(K):

        Y[k+1]=np.array([]).reshape(2,0) # populate our dictionary with k elements(representing our clusters), each having a default key and a 2,0 array

    

    ####IMPORTANT######

    #!!!!!!!!!!!!!!!#

    for i in range(m):

         Y[C[i]]=np.c_[Y[C[i]],X[i]]# addding all the datapoints to chosen clusters in Y, based on their stored indexes in C



    for k in range(K):

        Y[k+1]=Y[k+1].T #reshaping all the points to look like 2,0 arrays



    for k in range(K):

         Centroids[:,k]=np.mean(Y[k+1],axis=0) # assiginng the mean of each group as new centroids



    Output=Y

Output
color=['red','blue','green','cyan','magenta']

labels=['cluster1','cluster2','cluster3','cluster4','cluster5']



for k in range(K):

    plt.scatter(Output[k+1][:,0],Output[k+1][:,1],c=color[k],label=labels[k])

plt.scatter(Centroids[0,:],Centroids[1,:],s=300,c='yellow',label='Centroids')