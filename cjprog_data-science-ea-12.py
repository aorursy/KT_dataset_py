import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def plot_data_set_2d(ed_csv):
    x1 = ed_csv["V1"]
    x2 = ed_csv["V2"]
    plt.plot()
    plt.xlim([ed_csv["V1"].min(), ed_csv["V1"].max()])
    plt.ylim([ed_csv["V1"].min(), ed_csv["V2"].max()])
    plt.title('Dataset')
    plt.scatter(x1, x2)
    plt.show()


def plot_elbow_curve(K, distortions):
    # create new plot and data
    plt.plot()
    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

def plot_voronoi(km, data2D):
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = data2D[:, 0].min() - 1, data2D[:, 0].max() + 1
    y_min, y_max = data2D[:, 1].min() - 1, data2D[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = km.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(data2D[:, 0], data2D[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = km.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()


def evaluate_cluster(data_2d, num_clusters=2, range_evaluate=5):
    silhouette_avg = 0
    while num_clusters < range_evaluate:
        km = KMeans(n_clusters=num_clusters)
        cluster_labels = km.fit_predict(data_2d)
        silhouette = silhouette_score(data_2d, cluster_labels)
        if silhouette > silhouette_avg:
            silhouette_avg = silhouette
            num_clusters_ev = num_clusters
        print("Clusters: %d, silhouette_avg: %f" % (num_clusters, silhouette))
        num_clusters += 1
    return num_clusters_ev
    RANGE_TO_EVALUATE = 20
    X = ed_csv = pd.read_csv("../input/edlich-kmeans-A0.csv")
    print(ed_csv.head())
plot_data_set_2d(ed_csv)
pca = PCA(n_components=2).fit(X)
data2D = pca.transform(X)

k_cluster_optimum = evaluate_cluster(data2D, range_evaluate=RANGE_TO_EVALUATE)
print("K cluster optimum with silhouette score: %d" % k_cluster_optimum)
# k means determine k with elbow curve
distortions = []
K = range(1,RANGE_TO_EVALUATE)
for k in K:
    km = KMeans(n_clusters=k).fit(X)
    km.fit(X)
    distortions.append(sum(np.min(cdist(X, km.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

plot_elbow_curve(K, distortions)
km = KMeans(n_clusters=k_cluster_optimum).fit(X)
km.fit(data2D)
plot_voronoi(km, data2D)