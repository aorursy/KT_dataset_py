!pip install sklearn
import numpy as np

import matplotlib.pyplot as plt

from matplotlib import style

style.use("ggplot")

from sklearn.cluster import KMeans
x=[4,8,1.4,7,5,1]

y=[5,6,1,2.5,8,7]

plt.scatter(x,y)

plt.show()

x=np.array([[4,5],[8,6],[1.4,1],[7,2.5],[5,8],[1,7]])
kmeans = KMeans(n_clusters=2)

kmeans.fit(x)
centroids = kmeans.cluster_centers_

labels = kmeans.labels_
print(centroids)

print(labels)
colors = ["140","280"]
for i in range(len(X)):

    print("coordinate:", X[i], "label:", labels[i])

    plt.scatter(x,y, c=[colors[l_] for l_ in labels], label=labels)

    plt.scatter(centroids[:, 0],centroids[:, 1], c=[c for c in colors[:len(centroids)]], marker = "x", s=170, linewidths = 5, zorder = 10)

    plt.show()