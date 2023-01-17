import numpy as np

from scipy.spatial import distance as dis





def sum_distance(clusters, centers):

    sum_all = 0

    for i in range(len(centers)):

        sum_cluster = 0

        for j in range(len(clusters[i])):

            sum_cluster += np.sum(np.power(np.array(centers[i]) - np.array(clusters[i][j]), 2))

        sum_all += sum_cluster



    return sum_all





class KMeans():

    def __init__(self, n_clusters=2, max_iter=500):

        self.k = n_clusters

        self.max_iterations = max_iter

        self.kmeans_centroids = []



    # Initialize the centroids as random samples

    def _init_random_centroids(self, data):

        n_samples, n_features = np.shape(data)

        centroids = np.zeros((self.k, n_features))

        for i in range(self.k):

            centroid = data[np.random.choice(range(n_samples))]

            centroids[i] = centroid

        return centroids



    # Return the index of the closest centroid to the sample

    def _closest_centroid(self, sample, centroids):

        closest_i = None

        closest_distance = float("inf")

        for i, centroid in enumerate(centroids):

            distance = dis.euclidean(sample, centroid)

            if distance < closest_distance:

                closest_i = i

                closest_distance = distance

        return closest_i



    # Assign the samples to the closest centroids to create clusters

    def _create_clusters(self, centroids, data):

        clusters = [[] for _ in range(self.k)]

        for sample_i, sample in enumerate(data):

            centroid_i = self._closest_centroid(sample, centroids)

            clusters[centroid_i].append(sample_i)

        return clusters



    # Calculate new centroids as the means of the samples in each cluster

    def _calculate_centroids(self, clusters, data):

        n_samples, n_features = np.shape(data)

        centroids = np.zeros((self.k, n_features))

        for i, cluster in enumerate(clusters):

            # Here we handle null clusters

            if len(cluster) != 0:

                centroid = np.mean(data[cluster], axis=0)

            else:

                centroid = data[np.random.choice(range(n_samples))]

            centroids[i] = centroid

        return centroids



    # Classify samples as the index of their clusters

    def _get_cluster_labels(self, clusters, data):

        # One prediction for each sample

        y_pred = np.zeros(np.shape(data)[0])

        for cluster_i, cluster in enumerate(clusters):

            for sample_i in cluster:

                y_pred[sample_i] = cluster_i

        return y_pred



    # Do K-Means clustering and return the centroids of the clusters

    def fit(self, data):

        # Initialize centroids

        centroids = self._init_random_centroids(data)

        # Iterate until convergence or for max iterations

        for _ in range(self.max_iterations):

            # Assign samples to closest centroids (create clusters)

            clusters = self._create_clusters(centroids, data)



            prev_centroids = centroids

            # Calculate new centroids from the clusters

            centroids = self._calculate_centroids(clusters, data)



            # If no centroids have changed => convergence

            diff = centroids - prev_centroids

            if not diff.any():

                break

        self.kmeans_centroids = centroids

        return self



    # Predict the class of each sample

    def predict(self, data):



        # First check if we have determined the K-Means centroids

        if not self.kmeans_centroids.any():

            raise Exception("K-Means centroids have not yet been determined.\nRun the K-Means 'fit' function first.")



        clusters = self._create_clusters(self.kmeans_centroids, data)



        predicted_labels = self._get_cluster_labels(clusters, data)



        return predicted_labels
from sklearn.cluster import DBSCAN

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from scipy.spatial import distance
df = pd.read_csv('../input/clustering-algorithms-applied-to-covid19-dataset/Dataset1.csv')

n_cluster = 4

max_iter = 20
X = df.to_numpy()

kmeans = KMeans(n_cluster, max_iter).fit(X)

predict = kmeans.predict(X)

clusters = [[] for i in range(n_cluster)]

for i in range(len(predict)):

    clusters[int(predict[i])].append(X[i])
centers = kmeans.kmeans_centroids
for i in range(n_cluster):

    plt.plot(np.array(clusters[i])[:,0], np.array(clusters[i])[:,-1], 's',color=(1 - (i/n_cluster), 1 - (i/n_cluster), (i/n_cluster)))

    plt.plot(centers[i][0],centers[i][1], 'co')

plt.show()
errors = []

for i in range(n_cluster):

    length = len(clusters[i])

    error = 0

    for dot in clusters[i]:

        error += distance.euclidean(dot, centers[i])

    errors.append(error/length)

print(errors)
distorsions = []

for k in range(1, 15):

    kmeans = KMeans(n_clusters=k)

    kmeans.fit(X)

    predict = kmeans.predict(X)

    clusters = [[] for i in range(k)]

    centers = kmeans.kmeans_centroids

    for i in range(len(predict)):

        clusters[int(predict[i])].append(X[i])

    distorsions.append(sum_distance(np.array(clusters), np.array(centers)))



fig = plt.figure(figsize=(15, 5))

plt.plot(range(1, 15), distorsions)

plt.grid(True)

plt.title('Elbow curve')
df2 = pd.read_csv('../input/clustering-algorithms-applied-to-covid19-dataset/Dataset2.csv')

plt.plot(df2.to_numpy(), 'bo')

plt.show()
!pip3 install folium
import folium
m = folium.Map(location=[32.427910, 53.688046], zoom_start=5)
COVID19_df = pd.read_csv('../input/clustering-algorithms-applied-to-covid19-dataset/covid-sample.csv')

print(COVID19_df.to_numpy()[:,-1].max(), COVID19_df.to_numpy()[:,-1].min())

print(COVID19_df.to_numpy()[:,0].max(), COVID19_df.to_numpy()[:,0].min())
for item in COVID19_df.to_numpy():

    folium.Circle(location = item, radius= 1, color="red", fill=True).add_to(m)

m
eps = 0.1

min_samples = 10

cluster_number = 4

clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(COVID19_df.to_numpy())

clusters = np.array(clustering.labels_)

samples = []

clusters_count = np.unique(clusters).shape[0] - 2

dataset = COVID19_df.to_numpy()

m = folium.Map(location=[32.427910, 53.688046], zoom_start=5)

print("Total number of classes: ", clusters_count)

for i in range(len(dataset)):

    if(clusters[i] == cluster_number):

        samples.append(dataset[i])

for item in samples:

    folium.Circle(location = item, radius= 1, color="red", fill=True).add_to(m)

m
colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred','lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray', 'blue', 'green', 'purple', 'orange', 'darkred','lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'black']

m = folium.Map(location=[32.427910, 53.688046], zoom_start=5)

for i in range(-1, clusters_count):

    for j in range(len(dataset)):

        if i == clusters[j]:

            folium.CircleMarker(location = dataset[j], radius= 1, color=colors[i], fill=True).add_to(m)

m
from matplotlib import image

img = image.imread('../input/clustering-algorithms-applied-to-covid19-dataset/imageSmall.png')

plt.imshow(img)

plt.show()
rows = img.shape[0]

cols = img.shape[1]

img = img.reshape(img.shape[0]*img.shape[1],3)

kmeans = KMeans(40,200)

kmeans.fit(img)

labels_ = kmeans.predict(img)

clusters = np.asarray(kmeans.kmeans_centroids,dtype=np.uint8) 

labels = np.asarray(labels_,dtype=np.uint8 )  

labels = labels.reshape(rows,cols);

plt.imshow(labels)

plt.show()