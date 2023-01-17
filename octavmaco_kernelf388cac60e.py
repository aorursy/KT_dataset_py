import numpy as np

import matplotlib.pyplot as plt

from matplotlib import style

style.use("ggplot")

from sklearn.cluster import KMeans



#Trasarea si vizualizarea datelor inainte de a le prelucra cu algoritmul de invatare automata

x = [1, 5, 1.5, 8, 1, 9, 7, 2, 3, 6]

y = [2, 8, 1.8, 8, 0.6, 11, 9, 3, 2, 7]

plt.scatter(x,y)

plt.show()
# Conversia datelor intr-un vector de tip NumPyarray

X = np.array([[1, 2], [5, 8], [1.5, 1.8], [8, 8], [1, 0.6], [9, 11], [7, 9], [2, 3], [3, 2], [6, 7]])



#Initializarea algoritmului K-means cu parametrii necesara si utilizarea functie .fit() pentru calibrare

kmeans = KMeans(n_clusters=2)

kmeans.fit(X)



#Aratarea valorilor centroizilor si a etichetelor privind precizia modelului

centroids = kmeans.cluster_centers_

labels = kmeans.labels_

print(centroids)

print(labels)
colors = ["g.", "r.", "c.", "y."]

for i in range(len(x)):

    print("Coordonate:",X[i], "Eticheta:", labels[i])

    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)

plt.scatter(centroids[:, 0], centroids[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)

plt.show