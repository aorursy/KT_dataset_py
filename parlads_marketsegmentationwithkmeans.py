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
sns.set()

from sklearn.cluster import KMeans
data = pd.read_csv('/kaggle/input/market-segmentationcsv/3.12. Example.csv')
data
# plot the data

plt.scatter(data.Satisfaction, data.Loyalty)
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')

plt.show()
x= data.copy()
kmeans  =KMeans()
kmeans.fit(x)
cluster  = x.copy()
cluster['cluster_pred'] = kmeans.fit_predict(x)
# plot the data

plt.scatter(cluster.Satisfaction, cluster.Loyalty, c=cluster.cluster_pred, cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')

plt.show()
#standarizing satisfaction

from sklearn import preprocessing
x_scale = preprocessing.scale(x)
x_scale
wcss = []

for i in range(1,10):
    kmeans = KMeans(i)
    kmeans.fit(x_scale)
    wcss.append(kmeans.inertia_)
    
wcss
plt.plot(range(1,10), wcss)
plt.xlabel('Number of cluster')
plt.ylabel('wcss')

plt.show()
k_means_new = KMeans(4)
k_means_new.fit(x_scale)

cluster_new = x.copy()
cluster_new['cluster_pred'] = k_means_new.fit_predict(x_scale)
cluster_new
# plot the data

plt.scatter(cluster_new.Satisfaction, cluster_new.Loyalty, c=cluster_new.cluster_pred, cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')

plt.show()