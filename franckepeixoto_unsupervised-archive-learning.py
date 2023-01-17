import sys

import sklearn

import numpy as np

import os

np.random.seed(42)

import matplotlib as mpl

import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)

mpl.rc('xtick', labelsize=12)

mpl.rc('ytick', labelsize=12)

import warnings

warnings.filterwarnings(action="ignore", message="^internal gelsd")

from sklearn.datasets import load_iris

data =load_iris()

x = data.data

y = data.target

data.target_names
plt.figure(figsize=(9, 3.5))



plt.subplot(121)

plt.plot(x[y==0, 2], x[y==0, 3], "yo", label="Iris setosa")

plt.plot(x[y==1, 2], x[y==1, 3], "bs", label="Iris versicolor")

plt.plot(x[y==2, 2], x[y==2, 3], "g^", label="Iris virginica")

plt.xlabel("Petal length", fontsize=14)

plt.ylabel("Petal width", fontsize=14)

plt.legend(fontsize=12)



plt.subplot(122)

plt.scatter(x[:, 2], x[:, 3], c="k", marker=".")

plt.xlabel("Petal length", fontsize=14)

plt.tick_params(labelleft=False)
from sklearn.mixture import GaussianMixture

y_pred = GaussianMixture(n_components=3, random_state=42).fit(x).predict(x)

mapping = np.array([2, 0, 1])

y_pred = np.array([mapping[cluster_id] for cluster_id in y_pred])

plt.plot(x[y_pred==0, 2], x[y_pred==0, 3], "yo", label="Cluster 1")

plt.plot(x[y_pred==1, 2], x[y_pred==1, 3], "bs", label="Cluster 2")

plt.plot(x[y_pred==2, 2], x[y_pred==2, 3], "g^", label="Cluster 3")

plt.xlabel("Petal length", fontsize=14)

plt.ylabel("Petal width", fontsize=14)

plt.legend(loc="upper left", fontsize=12)
np.sum(y_pred==y)
np.sum(y_pred==y) / len(y_pred)
from sklearn.datasets import make_blobs

blob_centers = np.array(

    [[ 0.2,  2.3],

     [-1.5 ,  2.3],

     [-2.8,  1.8],

     [-2.8,  2.8],

     [-2.8,  1.3]])

blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])

x, y = make_blobs(n_samples=2000, centers=blob_centers,

                  cluster_std=blob_std, random_state=7)

len(y),y
def plot_clusters(X, y=None,s=1,p=None):

    plt.scatter(X[:, 0], X[:, 1], c=y, s=s)

    plt.xlabel("$x_1$", fontsize=14)

    plt.ylabel("$x_2$", fontsize=14, rotation=0)

    if p != None:

        plt.scatter(p[:, 0], p[:, 1], c="+", s=s)

plt.figure(figsize=(8, 4))

plot_clusters(x,y)

#fit e predict

from sklearn.cluster import KMeans

k = 5

kmeans = KMeans(n_clusters=k, random_state=42)

y_pred = kmeans.fit_predict(x)

y_pred is kmeans.labels_

plot_clusters(kmeans.cluster_centers_,y=None,s=100)

X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])

kmeans.predict(X_new)





def plot_data(X):

    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)



def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):

    if weights is not None:

        centroids = centroids[weights > weights.max() / 10]

    plt.scatter(centroids[:, 0], centroids[:, 1],

                marker='o', s=30, linewidths=8,

                color=circle_color, zorder=10, alpha=0.9)

    plt.scatter(centroids[:, 0], centroids[:, 1],

                marker='x', s=50, linewidths=50,

                color=cross_color, zorder=11, alpha=1)



def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,

                             show_xlabels=True, show_ylabels=True):

    mins = X.min(axis=0) - 0.1

    maxs = X.max(axis=0) + 0.1

    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),

                         np.linspace(mins[1], maxs[1], resolution))

    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)



    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),

                cmap="Pastel2")

    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),

                linewidths=1, colors='k')

    plot_data(X)

    if show_centroids:

        plot_centroids(clusterer.cluster_centers_)



    if show_xlabels:

        plt.xlabel("$x_1$", fontsize=14)

    else:

        plt.tick_params(labelbottom=False)

    if show_ylabels:

        plt.ylabel("$x_2$", fontsize=14, rotation=0)

    else:

        plt.tick_params(labelleft=False)
plt.figure(figsize=(8, 4))

plot_decision_boundaries(kmeans, x)
t =np.linalg.norm(np.tile(X_new, (1, k)).reshape(-1, k, 2) - kmeans.cluster_centers_, axis=2)

plt.figure(figsize=(8, 4))

plot_data(t)

kmeans_iter1 = KMeans(n_clusters=5, init="random", n_init=1,

                     algorithm="full", max_iter=1, random_state=1)

kmeans_iter2 = KMeans(n_clusters=5, init="random", n_init=1,

                     algorithm="full", max_iter=2, random_state=1)

kmeans_iter3 = KMeans(n_clusters=5, init="random", n_init=1,

                     algorithm="full", max_iter=3, random_state=1)

kmeans_iter1.fit(x)

kmeans_iter2.fit(x)

kmeans_iter3.fit(x)
plt.figure(figsize=(10, 8))

plt.subplot(321)

plot_data(x)

plot_centroids(kmeans_iter1.cluster_centers_, circle_color='r', cross_color='w')

plt.ylabel("$x_2$", fontsize=14, rotation=0)

plt.tick_params(labelbottom=False)

plt.title("Update the centroids (initially randomly)", fontsize=14)



plt.subplot(322)

plot_decision_boundaries(kmeans_iter1, x, show_xlabels=False, show_ylabels=False)

plt.title("Label the instances", fontsize=14)



plt.subplot(324)

plot_decision_boundaries(kmeans_iter2, x, show_xlabels=False, show_ylabels=False)



plt.subplot(325)

plot_decision_boundaries(kmeans_iter2,x, show_centroids=False)

plot_centroids(kmeans_iter3.cluster_centers_)



plt.subplot(326)

plot_decision_boundaries(kmeans_iter3, x, show_ylabels=False)



def plot_clusterer_comparison(clusterer1, clusterer2, X, title1=None, title2=None):

    clusterer1.fit(X)

    clusterer2.fit(X)



    plt.figure(figsize=(10, 3.2))



    plt.subplot(121)

    plot_decision_boundaries(clusterer1, X)

    if title1:

        plt.title(title1, fontsize=14)



    plt.subplot(122)

    plot_decision_boundaries(clusterer2, X, show_ylabels=False)

    if title2:

        plt.title(title2, fontsize=14)





  
kmeans_rnd_init1 = KMeans(n_clusters=5, init="random", n_init=1,

                         algorithm="full", random_state=11)

kmeans_rnd_init2 = KMeans(n_clusters=5, init="random", n_init=1,

                         algorithm="full", random_state=19)



plot_clusterer_comparison(

    kmeans_rnd_init1, 

    kmeans_rnd_init2, 

    x,

    "Solution 1", "Solution 2 (with a different random init)")

kmeans.inertia_
kmeans_rnd_init1.inertia_
kmeans_rnd_10_inits = KMeans(n_clusters=5, init="random", n_init=10,

                              algorithm="full", random_state=11)

kmeans_rnd_10_inits.fit(x)
plt.figure(figsize=(8, 4))

plot_decision_boundaries(kmeans_rnd_10_inits, x)
KMeans()

good_init = np.array([[-3, 3], [-3, 2], [-3, 1], [-1, 2], [0, 2]])

kmeans = KMeans(n_clusters=5, init=good_init, n_init=1, random_state=42)

kmeans.fit(x)

kmeans.inertia_
%timeit -n 50 KMeans(algorithm="elkan").fit(x)
%timeit -n 50 KMeans(algorithm="full").fit(x)
from sklearn.cluster import MiniBatchKMeans

minibatch_kmeans = MiniBatchKMeans(n_clusters=5, random_state=42)

minibatch_kmeans.fit(x)
import urllib

from sklearn.datasets import fetch_openml



mnist = fetch_openml('mnist_784', version=1)

mnist.target = mnist.target.astype(np.int64)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    mnist["data"], mnist["target"], random_state=42)
filename = "my_mnist.data"

X_mm = np.memmap(filename, dtype='float32', mode='write', shape=X_train.shape)

X_mm[:] = X_train
minibatch_kmeans = MiniBatchKMeans(n_clusters=10, batch_size=10, random_state=42)

minibatch_kmeans.fit(X_mm)
def load_next_batch(batch_size):

    return x[np.random.choice(len(x), batch_size, replace=False)]
kmeans_k3 = KMeans(n_clusters=3, random_state=42)

kmeans_k8 = KMeans(n_clusters=8, random_state=42)



plot_clusterer_comparison(kmeans_k3, kmeans_k8, x, "$k=3$", "$k=8$")
kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(x) for k in range(1, 10)]

inertias = [model.inertia_ for model in kmeans_per_k]
plt.figure(figsize=(8, 3.5))

plt.plot(range(1, 10), inertias, "ro-")

plt.xlabel("$k$", fontsize=14)

plt.ylabel("Inertia", fontsize=14)

plt.annotate('Cotovelo',

             xy=(4, inertias[3]),

             xytext=(0.55, 0.55),

             textcoords='figure fraction',

             fontsize=16,

             arrowprops=dict(facecolor='black', shrink=0.1)

            )

plt.axis([1, 8.5, 0, 1300])
plot_decision_boundaries(kmeans_per_k[4-1], x)
from sklearn.metrics import silhouette_score

silhouette_score(x, kmeans.labels_)
silhouette_scores = [silhouette_score(x, model.labels_)

                     for model in kmeans_per_k[1:]]

plt.figure(figsize=(8, 3))

plt.plot(range(2, 10), silhouette_scores, "go-")

plt.xlabel("$k$", fontsize=14)

plt.ylabel("Silhouette score", fontsize=14)

plt.axis([1.8, 8.5, 0.55, 0.7])

from sklearn.metrics import silhouette_samples

from matplotlib.ticker import FixedLocator, FixedFormatter

X= x

plt.figure(figsize=(11, 9))



for k in (3, 4, 5, 6):

    plt.subplot(2, 2, k - 2)

    

    y_pred = kmeans_per_k[k - 1].labels_

    silhouette_coefficients = silhouette_samples(X, y_pred)



    padding = len(X) // 30

    pos = padding

    ticks = []

    for i in range(k):

        coeffs = silhouette_coefficients[y_pred == i]

        coeffs.sort()



        color = mpl.cm.Spectral(i / k)

        plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs,

                          facecolor=color, edgecolor=color, alpha=0.7)

        ticks.append(pos + len(coeffs) // 2)

        pos += len(coeffs) + padding



    plt.gca().yaxis.set_major_locator(FixedLocator(ticks))

    plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))

    if k in (3, 5):

        plt.ylabel("Cluster")

    

    if k in (5, 6):

        plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.xlabel("Silhouette Coefficient")

    else:

        plt.tick_params(labelbottom=False)



    plt.axvline(x=silhouette_scores[k - 2], color="red", linestyle="--")

    plt.title("$k={}$".format(k), fontsize=16)

PROJECT_ROOT_DIR="./"

images_path = os.path.join(PROJECT_ROOT_DIR, "images", "unsupervised_learning")

os.makedirs(images_path, exist_ok=True)

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"

filename = "ladybug.png"

print("Downloading", filename)

url = DOWNLOAD_ROOT + "images/unsupervised_learning/" + filename

url ='https://blog.influx.com.br/storage/app/uploads/public/67d/18f/2fd/67d18f2fdf601e40e21e8dc4e70247ca62c6ac83.jpg'

urllib.request.urlretrieve(url, os.path.join(images_path, filename))

from matplotlib.image import imread

images_path

image = imread(os.path.join(images_path, filename))

image.shape



X = image.reshape(-1, 3)

kmeans = KMeans(n_clusters=8, random_state=42).fit(X)

segmented_img = kmeans.cluster_centers_[kmeans.labels_]

segmented_img = segmented_img.reshape(image.shape)
segmented_imgs = []

n_colors = (10, 8, 6, 4, 2)

for n_clusters in n_colors:

    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)

    segmented_img = kmeans.cluster_centers_[kmeans.labels_]

    segmented_imgs.append(segmented_img.reshape(image.shape))
plt.figure(figsize=(10,5))

plt.subplots_adjust(wspace=0.05, hspace=0.1)



plt.subplot(231)

plt.imshow(image)

plt.title("Original image")

plt.axis('off')



for idx, n_clusters in enumerate(n_colors):

    plt.subplot(232 + idx)

    plt.imshow(segmented_imgs[idx])

    plt.title("{} colors {}".format(n_clusters, idx.real))

     

    plt.axis('off')
from sklearn.datasets import load_digits

X_digits, y_digits = load_digits(return_X_y=True)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, random_state=42)
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)

log_reg.fit(X_train, y_train)
log_reg.score(X_test, y_test)
from sklearn.pipeline import Pipeline

pipeline = Pipeline([

    ("kmeans", KMeans(n_clusters=50, random_state=42)),

    ("log_reg", LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)),

])

pipeline.fit(X_train, y_train)

pipeline.score(X_test, y_test)
from sklearn.model_selection import GridSearchCV

param_grid = dict(kmeans__n_clusters=range(2, 10))

grid_clf = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)

grid_clf.fit(X_train, y_train)

grid_clf.best_params_