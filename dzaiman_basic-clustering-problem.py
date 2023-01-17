import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.cm as cm

from sklearn.cluster import KMeans, SpectralClustering

from sklearn.datasets.samples_generator import(make_blobs, make_circles, make_moons)

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import silhouette_samples, silhouette_score, pairwise_distances_argmin





df = pd.read_csv("../input/old-faithful/faithful.csv")

df.head()
fig = plt.figure(figsize = (12,6))

plt.subplot(1,2,1)

sns.distplot(df.waiting, bins = 10)



plt.subplot(1,2,2)

sns.distplot(df.eruptions, bins = 10)



plt.show()
plt.scatter(df.eruptions, df.waiting)

plt.xlabel('Eruption time (min)')

plt.ylabel('Waiting interval (min)')

plt.title('Ol\' Faithful Geyser Eruption')
elbow = []

x = StandardScaler().fit_transform(df)

for i in range(1,10):

    km = KMeans(n_clusters = i, max_iter = 20, random_state = 20)

    km.fit(x)

    elbow.append(km.inertia_)



#Plot cluster

plt.plot(range(1,10), elbow)

plt.xlabel('Num of cluster')

plt.title('Elbow Method')

plt.ylabel('WCSS')

plt.show()
df = df[['eruptions', 'waiting']]

x = StandardScaler().fit_transform(df)



#kmeans

km = KMeans(n_clusters = 2, max_iter = 20, random_state = 20)

km.fit(x)

#Plot cluster

kmCenter = km.cluster_centers_

fig, ax = plt.subplots(figsize = (6,6))

plt.scatter(x[km.labels_ == 0,0], x[km.labels_ == 0,1], c = 'green', label = 'Cluster 1')

plt.scatter(x[km.labels_ == 1,0], x[km.labels_ == 1,1], c = 'blue', label = 'Cluster 2')

plt.scatter(kmCenter[:, 0], kmCenter[:,1], c = 'r', label = 'Centroid', marker = '*')



plt.legend()

plt.xlim([-2, 2])

plt.ylim([-2, 2])

plt.xlabel('Eruption time in mins')

plt.ylabel('Waiting time to next eruption')

plt.title('Visualization of clustered data')

ax.set_aspect('equal')
# very long way to visualise k-means scatterplot

fig, ax = plt.subplots(2, 2, figsize = (12,12), sharex = True, sharey = True)



#kmeans

km = KMeans(n_clusters = 1, max_iter = 20, random_state = 20)

km.fit(x)

kmCenter = km.cluster_centers_

#Plot cluster

ax[0,0].set_title('K = 1')

ax[0,0].scatter(x[km.labels_ == 0,0], x[km.labels_ == 0,1], c = 'green', label = 'Cluster 1')

ax[0,0].scatter(kmCenter[:, 0], kmCenter[:,1], c = 'r', label = 'Centroid', marker = '*')



km = KMeans(n_clusters = 2, max_iter = 20, random_state = 20)

km.fit(x)

kmCenter = km.cluster_centers_

#Plot cluster

ax[0,1].set_title('K = 2')

ax[0,1].scatter(x[km.labels_ == 0,0], x[km.labels_ == 0,1], c = 'green', label = 'Cluster 1')

ax[0,1].scatter(x[km.labels_ == 1,0], x[km.labels_ == 1,1], c = 'blue', label = 'Cluster 2')

ax[0,1].scatter(kmCenter[:, 0], kmCenter[:,1], c = 'r', label = 'Centroid', marker = '*')



km = KMeans(n_clusters = 3, max_iter = 20, random_state = 20)

km.fit(x)

kmCenter = km.cluster_centers_

#Plot cluster

ax[1,0].set_title('K = 3')

ax[1,0].scatter(x[km.labels_ == 0,0], x[km.labels_ == 0,1], c = 'green', label = 'Cluster 1')

ax[1,0].scatter(x[km.labels_ == 1,0], x[km.labels_ == 1,1], c = 'blue', label = 'Cluster 2')

ax[1,0].scatter(x[km.labels_ == 2,0], x[km.labels_ == 2,1], c = 'yellow', label = 'Cluster 3')

ax[1,0].scatter(kmCenter[:, 0], kmCenter[:,1], c = 'r', label = 'Centroid', marker = '*')



km = KMeans(n_clusters = 4, max_iter = 20, random_state = 20)

km.fit(x)

kmCenter = km.cluster_centers_

#Plot cluster

ax[1,1].set_title('K = 4')

ax[1,1].scatter(x[km.labels_ == 0,0], x[km.labels_ == 0,1], c = 'green', label = 'Cluster 1')

ax[1,1].scatter(x[km.labels_ == 1,0], x[km.labels_ == 1,1], c = 'blue', label = 'Cluster 2')

ax[1,1].scatter(x[km.labels_ == 2,0], x[km.labels_ == 2,1], c = 'yellow', label = 'Cluster 3')

ax[1,1].scatter(x[km.labels_ == 3,0], x[km.labels_ == 3,1], c = 'purple', label = 'Cluster 4')

ax[1,1].scatter(kmCenter[:, 0], kmCenter[:,1], c = 'r', label = 'Centroid', marker = '*')





plt.xlim([-2, 2])

plt.ylim([-2, 2])

plt.xlabel('Eruption time in mins')

plt.ylabel('Waiting time to next eruption')

fig.suptitle('Visualization of clustered data')

# Conciese and eficient method of iterating through differenct k-means along with it's silhouette analysis

c = [2,3,4]

for n in c:

    fig, (ax, ay) = plt.subplots(1,2,figsize = (12,8))

    ax.set_xlim([-0.1, 1])

    cluster = KMeans(n_clusters = n, max_iter = 20, random_state = 20)

    clusterLabel = cluster.fit_predict(x)

    silAvg = silhouette_score(x, clusterLabel)

    print("K = ", n," average silhouette scoe : ", silAvg)

    sampleSilVal = silhouette_samples(x, clusterLabel)

    

    yLower = 10

    for i in range(n): 

        clusterSilVal = sampleSilVal[clusterLabel == i]

        clusterSilVal.sort()

        

        iClusterSize = clusterSilVal.shape[0]

        yUpper = yLower + iClusterSize

        

        color = cm.nipy_spectral(float(i) / n)

        ax.fill_betweenx(np.arange(yLower, yUpper), 0, clusterSilVal, facecolor = color, edgecolor = color, alpha = 0.7)

        

        ax.text(-0.05, yLower + 0.5 * iClusterSize, str(i))

        yLower = yUpper + 10

    ax.axvline(x = silAvg, color = 'red', linestyle = '--')

    

    colors = cm.nipy_spectral(clusterLabel.astype(float) / i)

    ay.scatter(x[:,0] ,x[:,1] ,c = colors, edgecolor='k')



    

    centers = cluster.cluster_centers_

    # Draw white circles at cluster centers

    ay.scatter(centers[:, 0], centers[:, 1], marker='o',c="white", alpha=1, s=400, edgecolor='k')



    for i, c in enumerate(centers):

        ay.scatter(c[0], c[1], marker='$%d$' % i, cmap = 'winter')



    ay.set_title("Ol' Faithful")

    ay.set_xlabel("Eruption Time")

    ay.set_ylabel("Waiting Time")



    

plt.show()