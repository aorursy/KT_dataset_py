# importing libraries

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.datasets.samples_generator import make_blobs

from sklearn.cluster import KMeans
# creating 800 random data points, with 5 random center with a standard deviation 

# X - is the feature

# y_true - is the possible label generated



X, y_true = make_blobs(n_samples = 800, centers=4, cluster_std=0.4)
for i in range(10):

    print(X[i], '\t', y_true[i])
# plotting the data

sns.scatterplot(X[:, 0], X[:, 1], hue=y_true, palette='rainbow')

plt.show()
sse = {} # Sum of squared error of samples to their closest cluster center.



for k in range(1, 10):

    model = KMeans(n_clusters=k, max_iter=1000)

    model.fit(X)

    sse[k] = model.inertia_ 



print(sse)
plt.figure()

plt.plot(list(sse.keys()), list(sse.values()))

plt.xlabel("Number of cluster")

plt.ylabel("SSE")

plt.show("Number of clusters vs Sum of squared error")

plt.show()
from yellowbrick.cluster import KElbowVisualizer



visualizer = KElbowVisualizer(model, k=(1,12))

visualizer.fit(X)        # Fit the data to the visualizer

visualizer.show()        # Finalize and render the figure