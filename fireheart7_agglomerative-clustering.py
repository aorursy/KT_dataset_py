! conda install -y gdown
! gdown --id 1yypMeJQFpLrFsO9QuQJPYZ59oe5erO_w
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from tqdm.notebook import tqdm

import sklearn
mall_data = pd.DataFrame(pd.read_csv("./Mall_Customers.csv"))



print("Shape of mall data = ", mall_data.shape)
mall_data.head()
X = mall_data.iloc[:, [3, 4]].values



print("Shape of X = ", X.shape)

print("Type(X) = ", type(X))
import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(X, method = "ward")) # minimize variance



plt.title("Dendrogram")

plt.xlabel("Customers")

plt.ylabel("Euclidean Distance")

plt.show()
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters = 5, affinity = "euclidean", linkage = "ward")

y_hc = hc.fit_predict(X)
print(y_hc)
plt.figure(figsize = (12, 8))

plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'green', label = 'cluster I')

plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'cluster II')

plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'brown', label = 'cluster III')

plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'black', label = 'cluster IV')

plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'violet', label = 'cluster V')

plt.title("Hierarchical Clustering On Mall Data", fontsize = 18)

plt.xlabel("x1 = Annual Income", fontsize = 16)

plt.ylabel("x2 = Spending Income", fontsize = 16)

plt.grid(True)

#plt.set_axisbelow(True)

# Turn on the minor TICKS, which are required for the minor GRID

plt.minorticks_on()

# Customize the major grid

plt.grid(which='major', linestyle='-', linewidth='1.0', color='grey')

# Customize the minor grid

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.legend()