# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d import axes3d

from scipy.spatial import distance

from copy import deepcopy



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv(r'../input/Mall_Customers.csv')

data.head()
# Let's see how our data looks like for males/females

plt.figure(figsize=(15,5))

plt.subplot(131)

sns.kdeplot(data[data['Gender'] == 'Male']['Annual Income (k$)'], shade=True, label='Male')

sns.kdeplot(data[data['Gender'] == 'Female']['Annual Income (k$)'], shade=True, label='Female')

plt.xlabel('Annual Income')



plt.subplot(132)

sns.kdeplot(data[data['Gender'] == 'Male']['Spending Score (1-100)'], shade=True, label='Male')

sns.kdeplot(data[data['Gender'] == 'Female']['Spending Score (1-100)'], shade=True, label='Female')

plt.xlabel('Spending Score')



plt.subplot(133)

sns.kdeplot(data[data['Gender'] == 'Male']['Age'], shade=True, label='Male')

sns.kdeplot(data[data['Gender'] == 'Female']['Age'], shade=True, label='Female')

plt.xlabel('Age')



plt.show()
sns.lmplot(data=data, x='Spending Score (1-100)', y='Annual Income (k$)', hue='Gender', fit_reg=False, legend=True, legend_out=True, height=7)

plt.show()
k=5

points = data[['Spending Score (1-100)', 'Annual Income (k$)']].values

centroids = data.sample(n=k)[['Spending Score (1-100)', 'Annual Income (k$)']].values

limits = [np.amin(points[:, 0]), np.amax(points[:, 0]), np.amin(points[:, 1]), np.amax(points[:, 1])]



plt.scatter(centroids[:, 0], centroids[:, 1])

plt.axis(limits)

plt.title('Initial Centroids')

plt.xlabel('Spending Score')

plt.ylabel('Annual Income')

plt.show()
centroids_old = np.zeros((centroids.shape))

while not np.array_equal(centroids, centroids_old):

    distance_matrix = distance.cdist(points, centroids, 'euclidean')

    labels = np.argmin(distance_matrix, axis=1)

    for j in range(k):

        centroids_old = deepcopy(centroids)

        cluster_points = [points[i] for i in range(len(points)) if labels[i] == j]

        centroids[j] = np.mean(cluster_points, axis=0)



plt.figure(figsize=(20, 10)) 

plt.subplot(121)

plt.scatter(centroids[:, 0], centroids[:, 1])

plt.axis(limits)

plt.title('Centroids')

plt.xlabel('Spending Score')

plt.ylabel('Annual Income')



plt.subplot(122)

plt.scatter(points[:, 0], points[:, 1], c=labels)

plt.title('Clusters')

plt.xlabel('Spending Score')

plt.ylabel('Annual Income')



plt.show()