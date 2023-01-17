import numpy as np

import pandas as pd 



#load existing dataset stored as csv file 

#dataset = pd.read_excel("../input/TestDataSet/newDataSet5.xlsx")



dataset = pd.read_csv("../input/dataset6/newDataSet6.csv")

## get columns' titles

print('Dataset contains on the following columns:')

titles = dataset.columns 

print(titles)
# Split Dataset into Data and target



print('\n\n*********Split Dataset into Data and target********************')



# first five rows

dsTarget = dataset['Category']

dsTrain = dataset[["ReceivedYear","Items","Byear","Location"]]

print("dsData \n",dsTrain.head())

print("dsTarget \n",dsTarget.head())
#********************KMeans******************************

from sklearn.cluster import KMeans

from matplotlib import pyplot as plt

cluster = KMeans(n_clusters = 5)

cluster.fit(dataset)

pred = cluster.labels_

print('Accuracy scored using k-means clustering: ', pred)

#plt.scatter(dataset[:, 0], dataset[:, 1])

#plt.scatter(dataset, dsTrain)

plt.scatter(cluster.cluster_centers_[:, 0], cluster.cluster_centers_[:, 1], s=300, c='red')

plt.show()

from sklearn import metrics

print('Accuracy scored using k-means clustering:')

metrics.adjusted_rand_score(dsTarget, pred)
from sklearn.preprocessing import normalize

data_scaled = normalize(dataset)

data_scaled = pd.DataFrame(data_scaled, columns=dataset.columns)

data_scaled.head()

import scipy.cluster.hierarchy as shc

plt.figure(figsize=(10, 7))  

plt.title("Dendrograms")  

dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
from sklearn.cluster import DBSCAN





from sklearn.preprocessing import StandardScaler

stscaler = StandardScaler().fit(dataset)

stTrain = stscaler.transform(dataset)

dbsc = DBSCAN(eps = .5, min_samples = 15).fit(stTrain)

labels = dbsc.labels_

core_samples = np.zeros_like(labels, dtype = bool)

core_samples[dbsc.core_sample_indices_] = True



print('Accuracy scored using DBSCAN clustering:')

metrics.adjusted_rand_score(dsTarget, labels)