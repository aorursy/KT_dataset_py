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
#importing libaries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans 
from sklearn.cluster import MeanShift, estimate_bandwidth
import seaborn as sns
import os
#Reading the data

df = pd.read_csv("/kaggle/input/mall-customer/Mall_Customers.csv")
#data Visualizing 
df.head()

# checking if there is any NULL data

df.isnull().any().any()
# GEtting the insides of the data
df.isnull().sum()
df.describe()
df.Genre.value_counts()
sns.countplot(x='Genre', data=df)
plt.title('gender density')
plt.show()
totalgenre = df.Genre.value_counts()
genrelabel = ['Male', 'Female']
plt.axis('equal') # For perfect circle
plt.pie(totalgenre, labels=genrelabel, radius=1.5, autopct='%0.2f%%', shadow=True, explode=[0, 0], startangle=45)
# radius increase the size, autopct for show percentage two decimal point
plt.title('Ratio of Male & Female')
plt.show() 
df['Age'].describe()
my_bins=10
# Histogram used by deafult 10 bins . bins like range.
arr=plt.hist(df['Age'],bins=my_bins, rwidth=0.95) 
plt.xlabel('Age Class')
plt.ylabel('Frequency')
plt.title('Age Class')
X = df.iloc[:, [3, 4]].values

# let's check the shape of x
print(X.shape)
scores = []
values = np.arange(2,12)

#Iterate through the defined range
for num_clusters in values:
  #Train the KMeans clustering model
  kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
  kmeans.fit(X)
  score = metrics.silhouette_score(X, kmeans.labels_,metric='euclidean', sample_size=len(X),random_state= 0)
  print("\n Number of clusters = ",num_clusters)
  print("Silhouette score = ",score)
  scores.append(score)

#Extract best score and optimal number of clusters
num_clusters = np.argmax(scores) + values[0]
print('\nOptimal number of clusters = ',num_clusters)

plt.plot(range(1,11),scores)
plt.title('The silhouette method', fontsize = 15)
plt.xlabel('No of clusters')
plt.ylabel('scores')
plt.show()
km = KMeans(num_clusters, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_means = km.fit_predict(X)

plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s = 100, c = 'red', label = 'cluster_1')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s = 100, c = 'orange', label = 'cluster_2')
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], s = 100, c = 'green', label = 'cluster_3')
plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1], s = 100, c = 'pink', label = 'cluster_4')
plt.scatter(X[y_means == 4, 0], X[y_means == 4, 1], s = 100, c = 'blue', label = 'cluster_5')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'black' , label = 'centeroid')

plt.style.use('fivethirtyeight')
plt.title('K Means Clustering', fontsize = 20)
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.grid()
plt.show()
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(num_clusters, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'cluster_1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'orange', label = 'cluster_2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'cluster_3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'pink', label = 'cluster_4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'blue', label = 'cluster_5')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'black' , label = 'centeroid')

plt.style.use('fivethirtyeight')
plt.title('Hierarchial Clustering', fontsize = 20)
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.grid()
plt.show()
from sklearn.cluster import MeanShift
ms = MeanShift(bandwidth=2)
ms.fit(X)
ms_y_pred = ms.predict(X)

plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s = 100, c = 'red', label = 'cluster_1')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s = 100, c = 'orange', label = 'cluster_2')
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], s = 100, c = 'green', label = 'cluster_3')
plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1], s = 100, c = 'pink', label = 'cluster_4')
plt.scatter(X[y_means == 4, 0], X[y_means == 4, 1], s = 100, c = 'blue', label = 'cluster_5')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 150, c = 'black' , label = 'centeroid')
plt.style.use('fivethirtyeight')
plt.title('Mean shift clustering', fontsize = 20)
plt.xlabel('Annual income')
plt.ylabel('Spending Score')
plt.legend()
plt.grid()
plt.show()
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=5)
gmm.fit(X)
gmm_y_pred = gmm.predict(X)

plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s = 100, c = 'red', label = 'cluster_1')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s = 100, c = 'orange', label = 'cluster_2')
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], s = 100, c = 'green', label = 'cluster_3')
plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1], s = 100, c = 'pink', label = 'cluster_4')
plt.scatter(X[y_means == 4, 0], X[y_means == 4, 1], s = 100, c = 'blue', label = 'cluster_5')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 150, c = 'black' , label = 'centeroid')

plt.style.use('fivethirtyeight')
plt.title('Gaussian mixture clustring', fontsize = 20)
plt.xlabel('Annual income')
plt.ylabel('Spending Score')
plt.legend()
plt.grid()
plt.show()
from sklearn.cluster import AffinityPropagation
model_aff = AffinityPropagation(damping=0.9)
model_aff.fit(X)
aff_y_pred = model_aff.predict(X)

plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s = 100, c = 'red', label = 'cluster_1')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s = 100, c = 'orange', label = 'cluster_2')
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], s = 100, c = 'green', label = 'cluster_3')
plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1], s = 100, c = 'pink', label = 'cluster_4')
plt.scatter(X[y_means == 4, 0], X[y_means == 4, 1], s = 100, c = 'blue', label = 'cluster_5')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 150, c = 'black' , label = 'centeroid')
plt.style.use('fivethirtyeight')
plt.title('Affininty_propagation', fontsize = 20)
plt.xlabel('Annual income')
plt.ylabel('Spending Score')
plt.legend()
plt.grid()
plt.show()
X = df.iloc[:, [2, 4]].values
X.shape
scores = []
values = np.arange(2,12)

#Iterate through the defined range
for num_clusters in values:
  #Train the KMeans clustering model
  kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
  kmeans.fit(X)
  score = metrics.silhouette_score(X, kmeans.labels_,metric='euclidean', sample_size=len(X),random_state= 0)
  print("\n Number of clusters = ",num_clusters)
  print("Silhouette score = ",score)
  scores.append(score)

#Extract best score and optimal number of clusters
num_clusters = np.argmax(scores) + values[0]
print('\nOptimal number of clusters = ',num_clusters)

plt.plot(range(1,11),scores)
plt.title('The silhouette method', fontsize = 15)
plt.xlabel('No of clusters')
plt.ylabel('scores')
plt.show()
km = KMeans(num_clusters, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_means = km.fit_predict(X)

plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s = 100, c = 'red', label = 'cluster_1')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s = 100, c = 'orange', label = 'cluster_2')
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], s = 100, c = 'green', label = 'cluster_3')
plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1], s = 100, c = 'pink', label = 'cluster_4')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'black' , label = 'centeroid')

plt.style.use('fivethirtyeight')
plt.title('K Means Clustering', fontsize = 20)
plt.xlabel('AGE')
plt.ylabel('Spending Score')
plt.legend()
plt.grid()
plt.show()
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(num_clusters, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'cluster_1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'orange', label = 'cluster_2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'cluster_3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'pink', label = 'cluster_4')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'black' , label = 'centeroid')

plt.style.use('fivethirtyeight')
plt.title('Hierarchial Clustering', fontsize = 20)
plt.xlabel('AGE')
plt.ylabel('Spending Score')
plt.legend()
plt.grid()
plt.show()
from sklearn.cluster import MeanShift
ms = MeanShift(bandwidth=2)
ms.fit(X)
ms_y_pred = ms.predict(X)

plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s = 100, c = 'red', label = 'cluster_1')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s = 100, c = 'orange', label = 'cluster_2')
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], s = 100, c = 'green', label = 'cluMeanshift clusteringster_3')
plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1], s = 100, c = 'pink', label = 'cluster_4')
plt.scatter(X[y_means == 4, 0], X[y_means == 4, 1], s = 100, c = 'blue', label = 'cluster_5')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 150, c = 'black' , label = 'centeroid')
plt.style.use('fivethirtyeight')
plt.title('Mean shift clustering', fontsize = 20)
plt.xlabel('Age')
plt.ylabel('Spending Score')
plt.legend()
plt.grid()
plt.show()
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=5)
gmm.fit(X)
gmm_y_pred = gmm.predict(X)

plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s = 100, c = 'red', label = 'cluster_1')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s = 100, c = 'orange', label = 'cluster_2')
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], s = 100, c = 'green', label = 'cluster_3')
plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1], s = 100, c = 'pink', label = 'cluster_4')
plt.scatter(X[y_means == 4, 0], X[y_means == 4, 1], s = 100, c = 'blue', label = 'cluster_5')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 150, c = 'black' , label = 'centeroid')

plt.style.use('fivethirtyeight')
plt.title('Gaussian mixture', fontsize = 20)
plt.xlabel('AGE')
plt.ylabel('Spending Score')
plt.legend()
plt.grid()
plt.show()
from sklearn.cluster import AffinityPropagation
model_aff = AffinityPropagation(damping=0.9)
model_aff.fit(X)
aff_y_pred = model_aff.predict(X)

plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s = 100, c = 'red', label = 'cluster_1')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s = 100, c = 'orange', label = 'cluster_2')
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], s = 100, c = 'green', label = 'cluster_3')
plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1], s = 100, c = 'pink', label = 'cluster_4')
plt.scatter(X[y_means == 4, 0], X[y_means == 4, 1], s = 100, c = 'blue', label = 'cluster_5')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 150, c = 'black' , label = 'centeroid')
plt.style.use('fivethirtyeight')
plt.title('Affininty_propagation', fontsize = 20)
plt.xlabel('AGE')
plt.ylabel('Spending Score')
plt.legend()
plt.grid()
plt.show()
from sklearn.preprocessing import LabelEncoder
number = LabelEncoder()
df['Genre'] = number.fit_transform(df['Genre'])
X = df.iloc[:, [1, 4]].values
X.shape
scores = []
values = np.arange(2,12)

#Iterate through the defined range
for num_clusters in values:
  #Train the KMeans clustering model
  kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
  kmeans.fit(X)
  score = metrics.silhouette_score(X, kmeans.labels_,metric='euclidean', sample_size=len(X),random_state= 0)
  print("\n Number of clusters = ",num_clusters)
  print("Silhouette score = ",score)
  scores.append(score)

#Extract best score and optimal number of clusters
num_clusters = np.argmax(scores) + values[0]
print('\nOptimal number of clusters = ',num_clusters)

plt.plot(range(1,11),scores)
plt.title('The silhouette method', fontsize = 15)
plt.xlabel('No of clusters')
plt.ylabel('scores')
plt.show()
km = KMeans(num_clusters, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_means = km.fit_predict(X)

plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s = 100, c = 'red', label = 'cluster_1')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s = 100, c = 'orange', label = 'cluster_2')
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], s = 100, c = 'green', label = 'cluster_3')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'black' , label = 'centeroid')

plt.style.use('fivethirtyeight')
plt.title('K Means Clustering', fontsize = 20)
plt.xlabel('Gender')
plt.ylabel('Spending Score')
plt.legend()
plt.grid()
plt.show()
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(num_clusters, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'cluster_1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'orange', label = 'cluster_2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'cluster_3')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'black' , label = 'centeroid')

plt.style.use('fivethirtyeight')
plt.title('Hierarchial Clustering', fontsize = 20)
plt.xlabel('Gender')
plt.ylabel('Spending Score')
plt.legend()
plt.grid()
plt.show()
from sklearn.cluster import MeanShift
ms = MeanShift(bandwidth=2)
ms.fit(X)
ms_y_pred = ms.predict(X)

plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s = 100, c = 'red', label = 'cluster_1')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s = 100, c = 'orange', label = 'cluster_2')
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], s = 100, c = 'green', label = 'cluster_3')
plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1], s = 100, c = 'pink', label = 'cluster_4')
plt.scatter(X[y_means == 4, 0], X[y_means == 4, 1], s = 100, c = 'blue', label = 'cluster_5')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 150, c = 'black' , label = 'centeroid')
plt.style.use('fivethirtyeight')
plt.title('Mean shift clustering', fontsize = 20)
plt.xlabel('GENDER')
plt.ylabel('Spending Score')
plt.legend()
plt.grid()
plt.show()
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=5)
gmm.fit(X)
gmm_y_pred = gmm.predict(X)

plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s = 100, c = 'red', label = 'cluster_1')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s = 100, c = 'orange', label = 'cluster_2')
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], s = 100, c = 'green', label = 'cluster_3')
plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1], s = 100, c = 'pink', label = 'cluster_4')
plt.scatter(X[y_means == 4, 0], X[y_means == 4, 1], s = 100, c = 'blue', label = 'cluster_5')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 150, c = 'black' , label = 'centeroid')

plt.style.use('fivethirtyeight')
plt.title('Gaussian mixture', fontsize = 20)
plt.xlabel('GENDER')
plt.ylabel('Spending Score')
plt.legend()
plt.grid()
plt.show()
from sklearn.cluster import AffinityPropagation
model_aff = AffinityPropagation(damping=0.9)
model_aff.fit(X)
aff_y_pred = model_aff.predict(X)

plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s = 100, c = 'red', label = 'cluster_1')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s = 100, c = 'orange', label = 'cluster_2')
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], s = 100, c = 'green', label = 'cluster_3')
plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1], s = 100, c = 'pink', label = 'cluster_4')
plt.scatter(X[y_means == 4, 0], X[y_means == 4, 1], s = 100, c = 'blue', label = 'cluster_5')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 150, c = 'black' , label = 'centeroid')
plt.style.use('fivethirtyeight')
plt.title('Affininty_propagation', fontsize = 20)
plt.xlabel('GENDER')
plt.ylabel('Spending Score')
plt.legend()
plt.grid()
plt.show()