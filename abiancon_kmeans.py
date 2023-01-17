# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score, calinski_harabasz_score

from yellowbrick.cluster import KElbowVisualizer
dataset = pd.read_csv("../input/mall-customers/Mall_Customers.csv")

#Considero solo due features (Annual income e Spending Score)

X= dataset.iloc[:, [3,4]].values
plt.scatter(X[:,0], X[:,1], c='yellow', marker='o', edgecolor='black', s=100)
km = KMeans()

visualizer = KElbowVisualizer(km, k = (1,10))

visualizer.fit(X)

visualizer.poof()
model = KMeans(n_clusters = 4)

model.fit(X)

print('Silhouette Score')

print(silhouette_score(X, model.labels_, metric = 'euclidean'))

print('Calinski_harabasz Score')

print(calinski_harabasz_score(X, model.labels_))
plt.scatter(X[:,0], X[:,1], c = model.labels_, cmap = 'jet' )

plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], c = 'red', marker = 's')
visualizer = KElbowVisualizer(model, k = (2,10), metric = 'silhouette', timings = False)

visualizer.fit(X)

visualizer.poof()
visualizer = KElbowVisualizer(model, k = (2,10), metric = 'calinski_harabasz', timings = False)

visualizer.fit(X)

visualizer.poof()