%matplotlib inline

from sklearn.datasets import load_digits

from sklearn.cluster import KMeans

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()  # for plot styling

from scipy.stats import mode
digits = load_digits()

digits.data.shape
kmeans = KMeans(n_clusters=10, random_state=0)

clusters = kmeans.fit_predict(digits.data)

kmeans.cluster_centers_.shape
fig, ax = plt.subplots(2, 5, figsize=(8, 3))

centers = kmeans.cluster_centers_.reshape(10, 8, 8)

for axi, center in zip(ax.flat, centers):

    axi.set(xticks=[], yticks=[])

    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
labels = np.zeros_like(clusters)

for i in range(10):

    mask = (clusters == i)

    labels[mask] = mode(digits.target[mask])[0]
from sklearn.metrics import accuracy_score

print(f"Accuracy for KMeams : {accuracy_score(digits.target, labels)}")
from sklearn.metrics import confusion_matrix



print(f"Confusion Matrix KMeans")



mat = confusion_matrix(digits.target, labels)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,

            xticklabels=digits.target_names,

            yticklabels=digits.target_names)

plt.xlabel('true label')

plt.ylabel('predicted label')

plt.show()