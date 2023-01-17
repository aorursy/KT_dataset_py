# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

plt.rcParams["figure.figsize"] = (12,8)

sns.set()
# create a dataset with four centers

from sklearn.datasets.samples_generator import make_blobs



X, _ = make_blobs(n_samples=500, centers=4, cluster_std=.8, random_state=0)

plt.scatter(X[:,0], X[:,1], s=20)
# train the cluster model

from sklearn.cluster import KMeans



kmeans = KMeans(n_clusters=4)

kmeans.fit(X)
# perform the classification

y = kmeans.predict(X)



# see how four categories (centers) have been found

y[:10]
#plot the dataset by categories, with centroids



plt.scatter(X[:,0], X[:,1], c=y, s=20, cmap="viridis")

centroids = kmeans.cluster_centers_

plt.scatter(centroids[:,0], centroids[:,1], c="red", alpha=0.5, s=120)
# to understand how many centroids the model has to be trained with,

# plot many models and check when the curve starts rising (look for the elbow)

ssd = {}



for k in range(1,10):

    kmeans = KMeans(init="k-means++", n_clusters=k)

    kmeans.fit(X)

    ssd[k] = kmeans.inertia_

plt.plot(list(ssd.keys()), list(ssd.values()), marker="o")

plt.xlabel("Number of clusters", fontsize=16)

plt.ylabel("quadratic distances sum", fontsize=16)

plt.show()



#see that the elbow here is for k = 4