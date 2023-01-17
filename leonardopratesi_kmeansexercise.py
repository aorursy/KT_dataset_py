

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.datasets import make_blobs



#I create some random clusters with make_blobs algorithm, i manage to know the Ground Truth

X, y = make_blobs(n_samples=150, n_features=2,

   centers=6, cluster_std=0.7,

   shuffle=True, random_state=0)



plt.scatter(

   X[:, 0], X[:, 1],

   c='white', marker='o',

   edgecolor='black', s=50

)

plt.show()
from sklearn.cluster import KMeans



km = KMeans( n_clusters=6, init='random', #set number of clusters to 3

           n_init=10, max_iter=300,  #maxximum iterations for a run and 10 runs 

            tol=1e-04, random_state=0) # tolerance

y_km = km.fit_predict(X)



for x in range(6):

    clustername = "cluster " + str(x)

    plt.scatter(

        X[y_km == x, 0], X[y_km == x, 1],

        s=50, c=np.random.rand(3,),

        marker='s', edgecolor='black',

        label= clustername)



# plot the centroids

plt.scatter(

    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],

    s=250, marker='o',

    c='red', edgecolor='black',

    label='centroids'

)

plt.legend(scatterpoints=1)

plt.show()
from sklearn.cluster import AgglomerativeClustering

hac = AgglomerativeClustering(n_clusters=4, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=None)

y_hac = hac.fit_predict(X)



# plot the centroids

for x in range(4):

    clustername = "cluster " + str(x)

    plt.scatter(

        X[y_hac == x, 0], X[y_hac == x, 1],

        s=50, c=np.random.rand(3,),

        marker='s', edgecolor='black',

        label= clustername)

Xa = []

Ya = []

import csv

with open('/kaggle/input/cluster/datasetcluster.txt','r') as f:

    for line in f:

        x, y = line.split()

        Xa.append(int(x))

        Ya.append(int(y))

        

XYa = list(zip(Xa, Ya));

#sprint(XYa)

plt.figure(num=None, figsize=(20, 20), dpi=80, facecolor='w', edgecolor='k')

plt.scatter(

        Xa, Ya,

        s=50, c=np.random.rand(3,),

        marker='.', edgecolor='black',

        label= clustername)



print(XYa[0][0])
nclust = 15

#for nclust in range(4,6):

km = KMeans( n_clusters=nclust, init='random', #set number of clusters to 3

           n_init=10, max_iter=300,  #maxximum iterations for a run and 10 runs 

            tol=1e-04, random_state=0) # tolerance

y_km = km.fit_predict(XYa)





    

plt.figure(num=None, figsize=(20, 20), dpi=80, facecolor='w', edgecolor='k')

XYa = np.array(XYa)

for x in range(nclust):

    clustername = "cluster " + str(x)

    plt.scatter(

        XYa[y_km == x, 0], XYa[y_km == x, 1],

        s=50, c=np.random.rand(3,),

        marker='.', edgecolor='black',

        label= clustername)



plt.scatter(

    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],

    s=250, marker='o',

    c='red', edgecolor='black',

    label='centroids'

)

plt.legend(scatterpoints=1)

plt.show()
# I compare the results obtained with the ground truth and evaluate the accuracy with PURITY PARAMETER

from collections import Counter

truth =[];

with open('/kaggle/input/ground-truth/s4-label.pa','r') as f:

    for line in f:

        truth.append(int(line))



start=0

end=0

for k in range(1,nclust): #iterate through each cluster label

    while (truth[end] == k):

            end+=1;

    most_common,num_most_common_real = Counter(truth[start:end]).most_common(1)[0]; #count total items in each cluster

    most_common,num_most_common_eval = Counter(km.labels_[start:end]).most_common(1)[0]; #count max items of the same label in each cluster    

    start=end;

    print("Purity cluster " + str(k) +": "+ str(num_most_common_eval/num_most_common_real))

    k +=1;
