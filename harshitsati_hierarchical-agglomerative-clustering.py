import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dir = "../input/customer/customers.csv"
dataset = pd.read_csv(dir) #reads the csv file in a pandas dataframe
#dataset.head()

X = dataset.iloc[:, [3, 4]].values # using values of 3rd and 4th column, since those are the two neutral topics 
import scipy.cluster.hierarchy as sch
dendrogrm = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distance')
plt.show()


from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)
# documentation: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
# Visualising the clusters
plt.figure(figsize=(10, 7))
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 50, c = 'blue', label = 'Cautious')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 50, c = 'orange', label = 'Mediocre/Standard')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 50, c = 'green', label = 'Imporant Targets')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 50, c = 'cyan', label = 'Not Important/Careless')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 50, c = 'magenta', label = 'Diligent')
plt.title('Clustering of customers')
plt.xlabel('Annual Income (k)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()