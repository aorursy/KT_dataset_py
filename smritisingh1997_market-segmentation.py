import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

from sklearn.cluster import KMeans



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/3.12. Example.csv')

data.head()
plt.scatter(data['Satisfaction'], data['Loyalty'])

plt.xlabel('Satisfaction')

plt.ylabel('Loyalty')

plt.show()
x = data.copy()
kmeans = KMeans(2)

kmeans.fit(x)
clusters = x.copy()

clusters['cluster_pred'] = kmeans.fit_predict(x)
plt.scatter(clusters['Satisfaction'], clusters['Loyalty'], c=clusters['cluster_pred'], cmap='rainbow')

plt.xlabel('Satisfaction')

plt.ylabel('Loyalty')

plt.show()
from sklearn import preprocessing

x_scaled = preprocessing.scale(x)

x_scaled
wcss = []

for i in range(1, 10):

    kmeans = KMeans(i)

    kmeans.fit(x_scaled)

    wcss.append(kmeans.inertia_)

    

wcss
number_cluster = range(1, 10)

plt.plot(number_cluster, wcss)

plt.xlabel('Number of Clusters')

plt.ylabel('WCSS')

plt.show()
# Just change the parameter passed to the KMeans(2, 3, 4, 5)

kmeans_new = KMeans(4)

kmeans_new.fit(x_scaled)

clusters_new =  x.copy()

clusters_new['cluster_pred'] = kmeans_new.fit_predict(x_scaled)

clusters_new.head()
plt.scatter(clusters_new['Satisfaction'], clusters_new['Loyalty'], c=clusters_new['cluster_pred'], cmap='rainbow')

plt.xlabel('Satisfaction')

plt.ylabel('Loyalty')

plt.show()