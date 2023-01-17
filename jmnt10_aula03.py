%matplotlib inline

import matplotlib



import numpy as np

import matplotlib.pyplot as plt



from sklearn.cluster import KMeans

from sklearn.datasets import make_blobs
n_samples = 1500

random_state = 170

X, y = make_blobs(n_samples=n_samples, random_state=random_state)



#Incorrect number of clusters 

y_pred = KMeans(n_clusters=2, random_state=random_state).fit_predict(X)
plt.figure(figsize=(12, 12))

plt.subplot(221)

plt.scatter(X[:, 0], X[:, 1], c=y_pred)

plt.xlabel("gasto no final de semana")

plt.ylabel("quantidade de dependentes")

plt.title("Número incorreto de clusters")
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]

X_aniso = np.dot(X, transformation)

y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_aniso)

plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred)

plt.title("Anisotropicly Distributed Blobs")

plt.xlabel("gasto no final de semana")

plt.ylabel("quantidade de dependentes")
# Different variance

X_varied, y_varied = make_blobs(n_samples=n_samples,

                                cluster_std=[1.0, 2.5, 0.5],

                                random_state=random_state)

y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_varied)



plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred)

plt.title("Unequal Variance")

# Unevenly sized blobs

X_filtered = np.vstack((X[y == 0][:500], X[y == 1][:100], X[y == 2][:10]))

y_pred = KMeans(n_clusters=3,

                random_state=random_state).fit_predict(X_filtered)



plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_pred)

plt.title("Unevenly Sized Blobs")
from sklearn.cluster import KMeans

from sklearn import metrics

from scipy.spatial.distance import cdist

import numpy as np

import matplotlib.pyplot as plt



x1 = np.array([3, 1, 1, 2, 1, 6, 6, 6, 5, 6, 7, 8, 9, 8, 9, 9, 8])

x2 = np.array([5, 4, 5, 6, 5, 8, 6, 7, 6, 7, 1, 2, 1, 2, 3, 2, 3])



plt.plot()

plt.xlim([0, 10])

plt.ylim([0, 10])

plt.title('Dataset')

plt.scatter(x1, x2)

plt.show()



# create new plot and data

plt.plot()

X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)

colors = ['b', 'g', 'r']

markers = ['o', 'v', 's']



# k means determine k

distortions = []

K = range(1,10)

for k in K:

    kmeanModel = KMeans(n_clusters=k).fit(X)

    kmeanModel.fit(X)

    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])



# Plot the elbow

plt.plot(K, distortions, 'bx-')

plt.xlabel('k')

plt.ylabel('Distortion')

plt.title('The Elbow Method showing the optimal k')

plt.show()


y_pred = KMeans(n_clusters=3).fit_predict(X)



plt.figure(figsize=(12, 12))

plt.subplot(221)

plt.scatter(X[:, 0], X[:, 1], c=y_pred)

plt.xlabel("gasto no final de semana")

plt.ylabel("quantidade de dependentes")

plt.title("Número incorreto de clusters")