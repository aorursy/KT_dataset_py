# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']  # column names

d_iris_orig = pd.read_csv(url, names=names)   # read using pandas
d_iris_orig
d_iris_orig.plot(x='sepal-length', y='sepal-width', style='o')
import seaborn as sns

import matplotlib.pyplot as plt
sns.scatterplot(x='sepal-length', y='sepal-width', hue='class', data=d_iris_orig)
sns.pairplot(hue='class', data=d_iris_orig)
d_iris_orig_unsupervised = d_iris_orig.drop('class', axis=1)

d_iris_orig_unsupervised
iris_unsupervised_array = d_iris_orig_unsupervised.values

iris_unsupervised_array
K = 3

import random

centroids = [random.choice(iris_unsupervised_array) for i in range(K)]

centroids
from scipy.spatial.distance import euclidean
# clusters = {0:[], 1:[], 2: []}

def assign_clusters(iris_unsupervised_array, centroids):

    clusters = {i:[] for i in range(len(centroids))}  # Initilaisin CLusters

    for flower in iris_unsupervised_array: # Iterating Dataset for each flower

        min_distance = 999999 # Initialising minmum distance

        centroid = -1 # Initialising closest centroid

        for i,c in enumerate(centroids): # Iterating though centroid

            distance = euclidean(flower, c) # Finding distance

            if distance < min_distance: # Checking which is smaller

                centroid = i # Assign centroid if smaller

                min_distance = distance # Setting minimum distance

        # or 

    #     centroid = np.argmin([euclidean(flower, c) for c in centroids])     

        clusters[centroid].append(flower)

    return clusters



clusters = assign_clusters(iris_unsupervised_array, centroids)

clusters
def get_next_centroids(clusters):

    new_centroids = [] # Initialise New Centroids

    for centroid in clusters: # Iterating thought each cluster

        new_centroids.append(np.array(clusters[centroid]).mean(axis=0)) # Finding the centroid. In numpy this is very easy, 

        # mean with axis=0 will take the mean of every column

    return new_centroids

reset_centroids(clusters)
def step_K_means(iris_unsupervised_array, centroids):

    clusters = assign_clusters(iris_unsupervised_array, centroids)

    return get_next_centroids(clusters)

step_K_means(iris_unsupervised_array, centroids)
from tqdm import tqdm

def K_means(iris_unsupervised_array, K, max_steps=10):

    centroids = [random.choice(iris_unsupervised_array) for i in range(K)] # Initialise Centroids

    for step in tqdm(range(max_steps)):

        centroids = step_K_means(iris_unsupervised_array, centroids)

    return centroids
final_centroids = K_means(iris_unsupervised_array, 3, 5000)

final_centroids
def get_cluster(flower):

    return np.argmin([euclidean(flower, c) for c in final_centroids])   
d_iris_orig_unsupervised = d_iris_orig_unsupervised.drop('cluster', axis=1)
d_iris_orig_unsupervised['cluster'] = d_iris_orig_unsupervised.apply(get_cluster, axis=1)

d_iris_orig_unsupervised
d_iris_orig_unsupervised
from sklearn.cluster import KMeans

kmeans = KMeans(3).fit(iris_unsupervised_array)

kmeans
kmeans.cluster_centers_
final_centroids