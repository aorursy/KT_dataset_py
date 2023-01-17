from part1_cleaning import get_clean_data

from part3_textblobdata import get_textblob_data

df1, df2, df3 = get_clean_data()

textblob_df1, textblob_df2, textblob_df3 = get_textblob_data(df1, df2, df3)
X = textblob_df1.iloc[:, -2:].values
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)

y_kmeans = kmeans.fit_predict(X)
import matplotlib.pyplot as plt

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 20, c = 'red', label = 'C1')

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 20, c = 'blue', label = 'C2')

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 20, c = 'green', label = 'C3')
# import scipy.cluster.hierarchy as sch

# d = sch.dendrogram(sch.linkage(X, method = 'ward'))
from sklearn.cluster import AgglomerativeClustering

hc3 = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')

y_hc3 = hc3.fit_predict(X)
plt.scatter(X[y_hc3 == 0, 0], X[y_hc3 == 0, 1], s = 20, c = 'red', label = 'C1')

plt.scatter(X[y_hc3 == 1, 0], X[y_hc3 == 1, 1], s = 20, c = 'blue', label = 'C2')

plt.scatter(X[y_hc3 == 2, 0], X[y_hc3 == 2, 1], s = 20, c = 'green', label = 'C3')
from sklearn.metrics import confusion_matrix, accuracy_score

print(confusion_matrix(y_kmeans, y_hc3))

print(accuracy_score(y_kmeans, y_hc3))
# reassign the values, which 2, 1, 0 being a positive sentiment, neutral sentiment, and negative sentiment, respectively

c_sentiments = [2 if y == 1 else 1 if y == 2 else 0 for y in y_kmeans]

c_sentiments[0:10]
# Saving textblob sentiments

# %store -r df1

# final_df1 = df1

# final_df1['tb_sentiment'] = c_sentiments

# %store final_df1
X = textblob_df2.iloc[:, -2:].values
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)

y_kmeans = kmeans.fit_predict(X)
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 20, c = 'red', label = 'C1')

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 20, c = 'blue', label = 'C2')

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 20, c = 'green', label = 'C3')
# The codelines below are for Hierarchical clustering

hc3 = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')

y_hc3 = hc3.fit_predict(X)
plt.scatter(X[y_hc3 == 0, 0], X[y_hc3 == 0, 1], s = 20, c = 'red', label = 'C1')

plt.scatter(X[y_hc3 == 1, 0], X[y_hc3 == 1, 1], s = 20, c = 'blue', label = 'C2')

plt.scatter(X[y_hc3 == 2, 0], X[y_hc3 == 2, 1], s = 20, c = 'green', label = 'C3')
# reassign the values, which 2, 1, 0 being a positive sentiment, neutral sentiment, and negative sentiment, respectively

r_sentiments = [2 if y == 2 else 1 if y == 0 else 0 for y in y_kmeans]

r_sentiments[0:10]
# Saving textblob sentiments

# %store -r df2

# final_df2 = df2

# final_df2['tb_sentiment'] = r_sentiments

# %store final_df2
%store -r textblob_df3

X = textblob_df3.iloc[:, -1:].values
!pip install jenkspy

import jenkspy

breaks = jenkspy.jenks_breaks(X, nb_class=3)

plt.hist(X, bins = 50)

for b in breaks:

    plt.vlines(b, ymin=0, ymax=11000)
breaks
g_sentiments = [0 if x <= breaks[1] else 2 if x >= breaks[2] else 1 for x in X]

g_sentiments[0:10]
# Saving textblob sentiments

# %store -r df3

# final_df3 = df3

# final_df3['tb_sentiment'] = g_sentiments

# %store final_df3