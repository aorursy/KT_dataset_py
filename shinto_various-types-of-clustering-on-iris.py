import pandas as pd

iris = pd.read_csv('../input/Iris.csv')

iris.shape
iris.head()
iris.dtypes
iris.Species.value_counts()
# Convert the 'Species' column to numeric

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le.fit(iris['Species'])

iris['Species'] = le.transform(iris['Species'])

iris.Species.value_counts()
# Convert into numpy matrix

iris_matrix = iris.drop(['Id','Species'], axis=1).as_matrix()
# Try K-Means with 3 clusters

from sklearn.cluster import KMeans

from sklearn.metrics import adjusted_rand_score



seed = 7

kmeans = KMeans(n_clusters=3, random_state=seed)

labels = kmeans.fit_predict(iris_matrix)

adj_rand_score = adjusted_rand_score(iris['Species'], labels)

adj_rand_score
# Checking Silhouette Score - Not cross checking with original labels.

from sklearn.metrics import silhouette_score

for n in range(2, 11):

    kmeans = KMeans(n_clusters = n, random_state=10)

    labels = kmeans.fit_predict(iris_matrix)

    silhouette_avg = silhouette_score(iris_matrix, labels, metric='euclidean')

    adj_rand_score = adjusted_rand_score(iris.Species, labels)

    print("n: %d, silhouette score: %f, adj rand score: %f" % (n, silhouette_avg, adj_rand_score))
# Hierarchical clustering

from sklearn.cluster import AgglomerativeClustering

for n in range(2,11):

    hcl = AgglomerativeClustering(n_clusters=n, affinity='euclidean', linkage='ward')

    labels = hcl.fit_predict(iris_matrix)

    silhouette_avg = silhouette_score(iris_matrix, labels, metric='euclidean')

    adj_rand_score = adjusted_rand_score(iris.Species, labels)

    print("n: %d, silhouette score: %f, adj rand score: %f" % (n, silhouette_avg, adj_rand_score))