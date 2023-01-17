from sklearn.datasets.samples_generator import make_blobs



X, y = make_blobs(n_samples=1000, centers=2, n_features=2,random_state = 123)

print(X.shape)



print(y)

print(X)
from sklearn.cluster import KMeans

import pandas as pd

kmeans = KMeans(n_clusters = 3)

kmeans.fit(X)



clust_labels = kmeans.predict(X)

kmeans = pd.DataFrame(clust_labels)
from sklearn.metrics import classification_report, confusion_matrix  

print(confusion_matrix(y, clust_labels))
import matplotlib.pyplot as plt 



fig = plt.figure()

ax = fig.add_subplot(111)

scatter = ax.scatter(X[:,0] , X[:,1] ,  c=kmeans[0],s=50)



ax.set_title('K-Means Clustering')

ax.set_xlabel('Feature 1')

ax.set_ylabel('Feature 2')

plt.colorbar(scatter)