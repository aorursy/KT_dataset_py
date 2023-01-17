# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for data plotting

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/k_means/K_Means/"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/k_means/K_Means/Mall_Customers.csv")
df.head()
X = df.iloc[:,[3,4]].values
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Numver of clusters')
plt.ylabel('WCSS')
plt.show()
kmeans =  KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)
y_kmeans
plt.scatter(X[y_kmeans == 0, 0],X[y_kmeans ==0, 1], s = 100, c = 'red', label = 'Careful' )
plt.scatter(X[y_kmeans == 1, 0],X[y_kmeans ==1, 1], s = 100, c = 'blue', label = 'Standard' )
plt.scatter(X[y_kmeans == 2, 0],X[y_kmeans ==2, 1], s = 100, c = 'green', label = 'Target' )
plt.scatter(X[y_kmeans == 3, 0],X[y_kmeans ==3, 1], s = 100, c = 'cyan', label = 'Careless' )
plt.scatter(X[y_kmeans == 4, 0],X[y_kmeans ==4, 1], s = 100, c = 'magenta', label = 'Sensible' )
plt.title("Cluster of Clients")
plt.xlabel("Annual income (k$)")
plt.ylabel("Spending score (1-100)")
plt.legend()
plt.show()
