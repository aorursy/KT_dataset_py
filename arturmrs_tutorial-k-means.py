%matplotlib inline

from sklearn.datasets import make_blobs

from sklearn.datasets import make_moons

from sklearn.datasets import load_digits



from sklearn.cluster import KMeans

from sklearn.cluster import SpectralClustering

from sklearn.cluster import AgglomerativeClustering

from sklearn.manifold import TSNE



from sklearn.metrics import silhouette_score

from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score

from sklearn.metrics import confusion_matrix



from scipy.stats import mode

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
plt.style.use('ggplot')
X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=0.3, random_state=3)

plt.scatter(X[:, 0], X[:, 1])
kmeans = KMeans(n_clusters=4)

kmeans.fit(X)

y_result = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], cmap='cividis', c=y_result)
center = kmeans.cluster_centers_



plt.scatter(X[:, 0], X[:, 1], cmap='cividis', c=y_result)

plt.scatter(center[:, 0], center[:, 1], cmap='cividis', c='black')
cov1 = [[1, 0],[0, 1]]

cov2 = [[1, 0.5],[0.5, 1]]
X_1 = np.random.multivariate_normal([0, 10], cov1, (100,))

X_2 = np.random.multivariate_normal([5, 15], cov2, (100,))

X_3 = np.random.multivariate_normal([5, 10], cov2, (100,))

X = np.append(np.append(X_1, X_2, axis=0), X_3, axis=0)



plt.scatter(X[:,0], X[:,1])
kmeans_3 = KMeans(n_clusters=3)

kmeans_3.fit(X)

y_result_3 = kmeans_3.predict(X)



kmeans_4 = KMeans(n_clusters=4)

kmeans_4.fit(X)

y_result_4 = kmeans_4.predict(X)
plt.figure(figsize=(15,5))

plt.subplot(1, 2, 1)

plt.scatter(X[:, 0], X[:, 1], cmap='cividis', c=y_result_3)

plt.subplot(1, 2, 2)

plt.scatter(X[:, 0], X[:, 1], cmap='cividis', c=y_result_4)
sil_3 = silhouette_score(X, y_result_3)

sil_4 = silhouette_score(X, y_result_4)

print("Silhouette score for 3 clusters: {:.4f} | Silhouette score for 4 clusters: {:.4f}".format(sil_3, sil_4))
shs = []

x = []

for i in range(2,11):

    kmeans_i = KMeans(n_clusters=i)

    kmeans_i.fit(X)

    y_result_i = kmeans_i.predict(X)

    x.append(i)

    shs.append(silhouette_score(X, y_result_i))

    

plt.plot(x, shs)

plt.xlabel("Silhouette score")

plt.ylabel("Número de clusters")
X, y = make_moons(n_samples=300, noise=0.05, random_state=3)

plt.scatter(X[:,0], X[:,1])
kmeans_moons = KMeans(n_clusters=2, n_init=20)

kmeans_moons.fit(X)

y_result_moons = kmeans_moons.predict(X)

plt.scatter(X[:, 0], X[:, 1], cmap='cividis', c=y_result_moons)
spc = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='kmeans')

y_spc = spc.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], cmap='viridis', c=y_spc)
X, y = make_moons(n_samples=300, noise=0.09, random_state=3)

plt.scatter(X[:,0], X[:,1])
spc = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='kmeans')

y_spc = spc.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], cmap='viridis', c=y_spc)
digits = load_digits()

kmeans = KMeans(n_clusters=10, random_state=3)

y_result = kmeans.fit_predict(digits.data)
kmeans.cluster_centers_.shape
fig, ax = plt.subplots(2,5,figsize=(8,3))

centers = kmeans.cluster_centers_.reshape(10,8,8)

for axi, center in zip(ax.flat, centers):

    axi.set(xticks=[], yticks=[])

    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
labels = np.zeros_like(y_result)

for i in range(10):

    mask = (y_result == i)

    labels[mask] = mode(digits.target[mask])[0]
acc = accuracy_score(digits.target, labels)



print("Accuracy score:  {}".format(acc))
mx = confusion_matrix(digits.target, labels).max()

plt.figure(figsize=(8,8))

sns.heatmap(confusion_matrix(digits.target, labels)/mx, cmap='viridis', annot=True, fmt='.2f')

plt.xlabel("Label predito")

plt.ylabel("Label verdadeiro")
spc = SpectralClustering(n_clusters=10, affinity='nearest_neighbors', assign_labels='kmeans')

y_result = spc.fit_predict(digits.data)

labels = np.zeros_like(y_result)



for i in range(10):

    mask = (y_result == i)

    labels[mask] = mode(digits.target[mask])[0]

    

acc = accuracy_score(digits.target, labels)



print("Accuracy score:  {}".format(acc))
mx = confusion_matrix(digits.target, labels).max()

plt.figure(figsize=(8,8))

sns.heatmap(confusion_matrix(digits.target, labels)/mx, cmap='viridis', annot=True, fmt='.2f')

plt.xlabel("Label predito")

plt.ylabel("Label verdadeiro")
agg = AgglomerativeClustering(n_clusters=10)

y_result = agg.fit_predict(digits.data)

labels = np.zeros_like(y_result)



for i in range(10):

    mask = (y_result == i)

    labels[mask] = mode(digits.target[mask])[0]

    

acc = accuracy_score(digits.target, labels)

print("Accuracy score:  {}".format(acc))
mx = confusion_matrix(digits.target, labels).max()

plt.figure(figsize=(8,8))

sns.heatmap(confusion_matrix(digits.target, labels)/mx, cmap='viridis', annot=True, fmt='.2f')

plt.xlabel("Label predito")

plt.ylabel("Label verdadeiro")
tsne = TSNE(init='pca', random_state=3)

d_proj = tsne.fit_transform(digits.data)

kmeans = KMeans(n_clusters=10, random_state=3)

y_result = kmeans.fit_predict(d_proj)



labels = np.zeros_like(y_result)



for i in range(10):

    mask = (y_result == i)

    labels[mask] = mode(digits.target[mask])[0]

    

acc = accuracy_score(digits.target, labels)

print("Accuracy score:  {}".format(acc))
mx = confusion_matrix(digits.target, labels).max()

plt.figure(figsize=(8,8))

sns.heatmap(confusion_matrix(digits.target, labels)/mx, cmap='viridis', annot=True, fmt='.2f')

plt.xlabel("Label predito")

plt.ylabel("Label verdadeiro")
X = digits.data

y = digits.target



tsne = TSNE(init='pca', random_state=3)

X_plot = tsne.fit_transform(X)

target_ids = range(len(digits.target_names))



plt.figure(figsize=(10, 10))

colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'

for i, c, l in zip(target_ids, colors, digits.target_names):

    plt.scatter(X_plot[y == i, 0], X_plot[y == i, 1], c=c, label=l)

plt.legend()

plt.show()