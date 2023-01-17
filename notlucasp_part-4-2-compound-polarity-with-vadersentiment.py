from part1_cleaning import get_clean_data

from part4_vaderdata import get_vader_data

df1, df2, df3 = get_clean_data()

vader_df1, vader_df2, vader_df3 = get_vader_data(df1, df2, df3)
from sklearn.cluster import KMeans

X = vader_df1.iloc[:, -2:].values

kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)

y_kmeans = kmeans.fit_predict(X)
import matplotlib.pyplot as plt

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 20, c = 'red', label = 'C1')

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 20, c = 'blue', label = 'C2')

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 20, c = 'green', label = 'C3')
from sklearn.cluster import AgglomerativeClustering

hc3 = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')

y_hc3 = hc3.fit_predict(X)
plt.scatter(X[y_hc3 == 0, 0], X[y_hc3 == 0, 1], s = 20, c = 'red', label = 'C1')

plt.scatter(X[y_hc3 == 1, 0], X[y_hc3 == 1, 1], s = 20, c = 'blue', label = 'C2')

plt.scatter(X[y_hc3 == 2, 0], X[y_hc3 == 2, 1], s = 20, c = 'green', label = 'C3')
c_sentiments = [2 if y == 0 else 1 if y == 1 else 0 for y in y_hc3]

c_sentiments[0:10]
X = vader_df2.iloc[:, -2:].values

kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 1)

y_kmeans = kmeans.fit_predict(X)
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 20, c = 'red', label = 'C1')

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 20, c = 'blue', label = 'C2')

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 20, c = 'green', label = 'C3')
hc3 = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')

y_hc3 = hc3.fit_predict(X)
plt.scatter(X[y_hc3 == 0, 0], X[y_hc3 == 0, 1], s = 20, c = 'red', label = 'C1')

plt.scatter(X[y_hc3 == 1, 0], X[y_hc3 == 1, 1], s = 20, c = 'blue', label = 'C2')

plt.scatter(X[y_hc3 == 2, 0], X[y_hc3 == 2, 1], s = 20, c = 'green', label = 'C3')
# reassign the values, which 2, 1, 0 being a positive sentiment, neutral sentiment, and negative sentiment, respectively

r_sentiments = [2 if y == 1 else 1 if y == 2 else 0 for y in y_kmeans]

r_sentiments[0:10]
!pip install jenkspy

import jenkspy

X = vader_df3.iloc[:, -1].values

breaks = jenkspy.jenks_breaks(X, nb_class=3)

plt.hist(X, bins = 50)

for b in breaks:

    plt.vlines(b, ymin=0, ymax=11000)
breaks
g_sentiments = [0 if x <= breaks[1] else 2 if x >= breaks[2] else 1 for x in X]

g_sentiments[0:10]