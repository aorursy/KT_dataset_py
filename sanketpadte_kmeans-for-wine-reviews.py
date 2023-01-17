import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.cluster import KMeans



dataset = pd.read_csv('../input/winemag-data_first150k.csv',low_memory = False, encoding='ISO-8859-1')

df=dataset.dropna()

X = df.iloc[:, [10,4]].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()

X[:,0]=labelencoder_X.fit_transform(X[:,0])



import numpy as np

X[np.isnan(X)] = 0



from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X = sc_X.fit_transform(X)

from sklearn.cluster import KMeans

wcss = []

for i in range(1, 6):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

plt.plot(range(1, 6), wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
kmeans = KMeans(n_clusters =3, init = 'k-means++', random_state = 42)

y_kmeans = kmeans.fit_predict(X)



# Visualising the clusters

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')







plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')

plt.title('Wine Reviews')

plt.xlabel('Winery')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()