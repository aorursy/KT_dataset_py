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
from sklearn import datasets
myiris = datasets.load_iris()
x = myiris.data
y = myiris.target
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(x)
x_scaled = scaler.transform(x)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3, init = 'random', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x_scaled)
y_kmeans
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 50, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x_scaled)
y_kmeans
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = 3, init = 'random', max_iter = 50*i, n_init = 10, random_state = 0)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)
wcss
import matplotlib.pyplot as plt
plt.plot(max_iter, wcss)
plt.title('Graph')
plt.xlabel('max_iter')
plt.ylabel('WCSS')      #within cluster sum of squares
plt.show()