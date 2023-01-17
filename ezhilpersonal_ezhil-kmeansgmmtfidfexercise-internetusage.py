import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
df = pd.read_csv("/kaggle/input/advertising/advertising.csv")
df.head()
Sample_X= df[["Daily Time Spent on Site","Age","Area Income","Daily Internet Usage","Male"]]
target_y= df["Clicked on Ad"]
Sample_X.head()
target_y.head()
ss= StandardScaler()
ss.fit(Sample_X)
X = ss.transform(Sample_X)
kmeans = KMeans(n_clusters=2,
                    n_init =10,
                    max_iter = 800)
kmeans.fit(X)

centroids=kmeans.cluster_centers_

fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, s=2)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, linewidths=150, color='red')
plt.show()
kmean_pred = pd.DataFrame(kmeans.labels_,columns=["x"])
kmean_pred['y'] = target_y
#kmean_pred.head()
np.sum(kmean_pred["x"] == kmean_pred["y"])/len(kmean_pred["x"]) * 100
gm = GaussianMixture(
                     n_components = 2,
                     n_init = 10,
                     max_iter = 100)
gm.fit(X)

fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=gm.predict(X),s=5)
plt.scatter(gm.means_[:, 0], gm.means_[:, 1], marker='v', s=10, linewidths=5, color='red')
plt.show()
gmm_pred = pd.DataFrame(gm.predict(X),columns=["x"])
gmm_pred['y'] = target_y
np.sum(gmm_pred["x"] == gmm_pred["y"])/len(gmm_pred["x"]) * 100
tsne = TSNE(n_components = 2)
tsne_out = tsne.fit_transform(X)
plt.scatter(tsne_out[:, 0], tsne_out[:, 1], marker='x', s=50, linewidths=5, c=gm.predict(X))
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words = 'english')
trans_df = tfidf.fit_transform(df['Ad Topic Line'].values)
trans_columns = tfidf.get_feature_names()
trans_data = pd.DataFrame(trans_df.toarray(), columns = trans_columns                         )
trans_data.head()
scaled_5cols = pd.DataFrame(X, columns=Sample_X.columns )
final_df =  pd.concat([scaled_5cols,trans_data], axis =1)
final_df.head()
kmeans = KMeans(n_clusters=2,
                    n_init =10,
                    max_iter = 800)
kmeans.fit(final_df)

centroids=kmeans.cluster_centers_

fig = plt.figure()
plt.scatter(final_df.iloc[:, 0],final_df.iloc[:, 1], c=kmeans.labels_, s=2)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, linewidths=150, color='red')
plt.show()
kmean_pred = pd.DataFrame(kmeans.labels_,columns=["x"])
kmean_pred['y'] = target_y
np.sum(kmean_pred["x"] == kmean_pred["y"])/len(kmean_pred["x"]) * 100
gm = GaussianMixture(
                     n_components = 2,
                     n_init = 10,
                     max_iter = 100)
gm.fit(final_df)

fig = plt.figure()
plt.scatter(final_df.iloc[:, 0], final_df.iloc[:, 1], c=gm.predict(final_df),s=5)
plt.scatter(gm.means_[:, 0], gm.means_[:, 1], marker='v', s=10, linewidths=5, color='red')
plt.show()
gmm_pred = pd.DataFrame(gm.predict(final_df),columns=["x"])
gmm_pred['y'] = target_y
np.sum(gmm_pred["x"] == gmm_pred["y"])/len(gmm_pred["x"]) * 100
tsne = TSNE(n_components = 2)
tsne_out = tsne.fit_transform(final_df)
plt.scatter(tsne_out[:, 0], tsne_out[:, 1], marker='x', s=50, linewidths=5, c=gm.predict(final_df))