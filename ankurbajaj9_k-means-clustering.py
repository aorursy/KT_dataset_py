import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
!wget https://github.com/itisjoshi/ml/raw/master/data_1024.csv



drivers_data = pd.read_csv('data_1024.csv', sep='\t')



drivers_data.head(10)
drivers_data.shape
drivers_data = drivers_data.sample(frac=1)
drivers_data.drop('Driver_ID', axis=1, inplace=True)



drivers_data.sample(10)
fig, ax = plt.subplots(figsize=(10, 8))



plt.scatter(drivers_data['Distance_Feature'], 

            drivers_data['Speeding_Feature'], 

            s = 300, 

            c='blue')



plt.xlabel('Distance_Feature')

plt.ylabel('Speeding_Feature')



plt.show()
from sklearn.cluster import KMeans



#kmeans_model = KMeans(n_clusters=4, max_iter=1000).fit(drivers_data)



## For recording start with 4, then 3 then 2 and change this code and hit shift-enter

kmeans_model = KMeans(n_clusters=3, max_iter=1000).fit(drivers_data)

# kmeans_model = KMeans(n_clusters=2, max_iter=1000).fit(drivers_data)
kmeans_model.labels_[::40]
np.unique(kmeans_model.labels_)
zipped_list = list(zip(np.array(drivers_data), kmeans_model.labels_))



zipped_list[1000:1010]
centroids = kmeans_model.cluster_centers_



centroids
colors = ['g', 'y', 'b', 'k']

    

plt.figure(figsize=(10, 8))



for element in zipped_list:

    plt.scatter(element[0][0], element[0][1], 

                c=colors[(element[1] % len(colors))])

    

plt.scatter(centroids[:,0], centroids[:,1], c='r', s=200, marker='s')



for i in range(len(centroids)):

    plt.annotate( i, (centroids[i][0], centroids[i][1]), fontsize=20)
from sklearn.metrics import silhouette_score



print("Silhouette score: ", silhouette_score(drivers_data, kmeans_model.labels_))