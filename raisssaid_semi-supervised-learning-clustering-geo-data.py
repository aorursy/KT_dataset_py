import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

import folium
import re

# silhouette_score To select the appropriate number of clusters
from sklearn.metrics import silhouette_score
# kneed will return the knee point of the function.
# The knee point is the point of maximum curvature
from kneed import KneeLocator
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('../input/taxi-data/taxi_data.csv')
data.info()
# Let's check if there are any duplicates
duplicatedv = data.duplicated(subset = ['LON', 'LAT']).values.sum()
print('{:d} duplicate values'.format(duplicatedv))
# Remove duplicates and null values
data.drop_duplicates(
    subset=['LON','LAT'],
    keep='first',
    inplace=True
)
data.dropna(inplace=True)
# Let's check duplicates and null values
duplicatedv_ = data.duplicated(subset = ['LON', 'LAT']).values.sum()
nullv_ = data.isna().values.sum()
print('{:d} valeurs dupliquées'.format(duplicatedv_))
print('{:d} valeurs nulles'.format(nullv_))
data.info()
data.head()
X = data[['LAT', 'LON']].values
map = folium.Map(
    location = [X[:, 0].mean(), X[:, 1].mean()],
    zoom_start = 10
)
for _, row in data.iterrows():
  folium.CircleMarker(
      location = [row.LAT, row.LON],
      radius = 5,
      popup = re.sub(r'\W+', '', row.NAME),
      fill = True,
  ).add_to(map)
map
# Choose the best number of clusters using silhouette method
score = -1
scores = []
k = 0
for i in range(2, 101):
  kmeans = KMeans(n_clusters=i, random_state=0).fit(X)
  score_ = silhouette_score(X, kmeans.predict(X))
  scores.append(score_)
  if score_ > score:
    score = score_
    k=i
print(
'Best number of clusters is {} with a score of {:.2f}'
.format(k,score)
)
plt.plot(range(2, 101), scores)
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.xticks(np.arange(0, 101, 5))
plt.show()
# Choose the best number of clusters using Elbow method
inertia = []
for i in range(1, 101):
  kmeans = KMeans(n_clusters=i, random_state=0).fit(X)
  inertia.append(kmeans.inertia_)
plt.plot(range(1, 101), inertia)
plt.title('Elbow')
plt.xlabel('Number of clusters')
plt.ylabel('Inertie')
plt.xticks(np.arange(0, 101, 5))
plt.show()
knee_point = KneeLocator(
    range(1, 101),
    inertia,
    curve = 'convex',
    direction = 'decreasing'
)
print(knee_point.knee)
kmeans = KMeans(n_clusters=11, random_state=0).fit(X)
inertia_ = kmeans.inertia_
silhouette_ = silhouette_score(X, kmeans.labels_)
print("This model with 11 cluster is characterized by :")
# Iinertia is sum of squared distances of samples to their closest cluster center.
print("Iinertia : {:.3f}".format(inertia_))
print("Silhouette mean score : {:.3f}".format(silhouette_))
data['Cluster_KMeans'] = kmeans.labels_
data.head()
# Show results
colors = ['royalblue', 'maroon', 'forestgreen', 'mediumorchid', 'tan',
          'deeppink', 'olive', 'goldenrod', 'lightcyan', 'navy',
          'springgreen','midnightblue', 'red','brown','limegreen','lime',
          'pink','orchid','crimson','m']*10
vectorizer = np.vectorize(lambda x: colors[x % len(colors)])
data['Colors_KMeans'] = vectorizer(kmeans.labels_)
def create_map(cluster_column, colors_column, title):
    map = folium.Map(location=[X[:,0].mean(),X[:,1].mean()],zoom_start=8.5)
    for _,row in data.iterrows():
        folium.CircleMarker(
            location=[row.LAT,row.LON],
            radius=5,
            popup = re.sub(r'\W+', '', row.NAME),
            fill=True,
            color=row[colors_column],
            fill_color=row[colors_column],
        ).add_to(map)

    print(title)
    return map
create_map('Cluster_KMeans','Colors_KMeans','KMeans Clustering')
nearest_neighbors = NearestNeighbors(n_neighbors=7)
neighbors = nearest_neighbors.fit(X)
distances, indices = neighbors.kneighbors(X)
distances = np.sort(distances[:, 5], axis=0)
i = np.arange(len(distances))
knee = KneeLocator(
    i, distances, S=1,
    curve='convex',
    direction='increasing',
    interp_method='polynomial'
)
fig = plt.figure(figsize=(5, 5))
knee.plot_knee()
plt.xlabel("Points")
plt.ylabel("Distance")
plt.show()
EPS = round(distances[knee.knee],3)
print('The elbow point is at around {}'.format(EPS))
dbscan = DBSCAN(eps = EPS)
dbscan.fit(X)
dbscan_predictions = dbscan.labels_
data['Cluster_DBSCAN'] = dbscan_predictions
vectorizer = np.vectorize(
    lambda x: colors[x % len(colors)]
)
data['Colors_DBSCAN'] = vectorizer(dbscan_predictions)
create_map('Cluster_DBSCAN','Colors_DBSCAN','DBSCAN eps={}'.format(EPS))
round(silhouette_score(X,dbscan_predictions),2)
n_of_clusters = len(np.unique(dbscan_predictions))
n_of_outliers = len(dbscan_predictions[dbscan_predictions==-1])
print('Number of clusters is: {}'.format(n_of_clusters))
print('Number of outliers is: {}'.format(n_of_outliers))
outliers = [
    (counter+2)*x if x == -1 else x
    for counter,x in enumerate(dbscan_predictions)
]

ign_outliers_score = silhouette_score(
    X[dbscan_predictions != -1],
    dbscan_predictions[dbscan_predictions != -1]
)
outliers_singl_score = silhouette_score(X,outliers)
                                                                       
print(
    'Silhouette score without outliers: {:.2f}'
    .format(ign_outliers_score)
    )
print(
    'Silhouette score with outliers: {:.2f}'
    .format(outliers_singl_score)
    )
scores_no_outlier = []
scores_with_outlier = []
max_score = 0
best_eps = 0
for i in np.arange(0.15, 0, -0.005):
  dbscan = DBSCAN(eps = i)
  dbscan.fit(X)
  dbscan_predictions = dbscan.labels_
  score_without_outlier = silhouette_score(
      X[dbscan_predictions != -1],
      dbscan_predictions[dbscan_predictions != -1]
  )
  scores_no_outlier.append(score_without_outlier)
  outliers = [
              (counter+2)*x if x==-1 else x 
              for counter,x in enumerate(dbscan_predictions)
              ]
  scores_with_outlier.append(silhouette_score(X,outliers))
  if score_without_outlier > max_score:
        max_score = score_without_outlier
        best_eps = i
plt.figure(figsize=(10,6))
plt.plot(np.arange(0.15,0,-0.005),scores_no_outlier)
plt.xlabel('Epsilon')
plt.ylabel('Silhouette Score')
plt.title('Scores without outliers')
plt.show()
print(
    'Highest score = {} obtained for epsilon = {}'
    .format(round(max_score,3),round(best_eps,3))
)
dbscan = DBSCAN(eps=0.01)
dbscan.fit(X)
dbscan_predictions = dbscan.labels_
data['Cluster_DBSCAN_OPT'] = dbscan_predictions
vectorizer = np.vectorize(lambda x: colors[x % len(colors)])
data['Colors_DBSCAN_OPT'] = vectorizer(dbscan_predictions)
create_map('Cluster_DBSCAN_OPT','Colors_DBSCAN_OPT','DBSCAN OPT')
print('Number of clusters: {}'
  .format(len(np.unique(dbscan_predictions))))
print('Number of outliers: {}'
  .format(len(dbscan_predictions[dbscan_predictions==-1])))
outliers=[
          (counter+2)*x if x==-1 else x 
          for counter,x in enumerate(dbscan_predictions)]
print('Silhouette score without outliers: {}'
  .format(
      silhouette_score(X[dbscan_predictions!=-1],
      dbscan_predictions[dbscan_predictions!=-1])))
print('Silhouette score with outliers: {}'
  .format(silhouette_score(X,outliers)))
X_train = data[data['Cluster_DBSCAN_OPT'] != -1][['LON', 'LAT']]
y_train = data[data['Cluster_DBSCAN_OPT'] != -1][['Cluster_DBSCAN_OPT']]
X_pred = data[data['Cluster_DBSCAN_OPT'] == -1][['LON', 'LAT']]
print('Dimensions matrix: X_train ')
print(X_train.head())
print('Labels vector: y_train')
print(y_train.head())
print('Samples to be classified: X_pred')
print(X_pred.head())
# Before we use the model, let us first check
# the best number of neighbors to use
scores = []
for i in range(1, 11):
  knc = KNeighborsClassifier(n_neighbors=i)
  knc.fit(X_train, y_train.values.ravel())
  scores.append(knc.score(X_train, y_train))
plt.plot(range(1, 11), scores)
plt.xticks(np.arange(1, 11, 4))
plt.show()
KNC = KNeighborsClassifier(n_neighbors=5)
KNC.fit(X_train, y_train.values.ravel())
KNC_predictions = KNC.predict(X_pred)
data['Cluster_Hybrid'] = data['Cluster_DBSCAN_OPT']
data.loc[data['Cluster_DBSCAN_OPT'] == -1, 'Cluster_Hybrid'] = KNC_predictions
vectorizer = np.vectorize(lambda x: colors[x % len(colors)])
data['Colors_Hybrid'] = vectorizer(data['Cluster_Hybrid'].values)
create_map('Cluster_Hybrid','Colors_Hybrid','Hybrid (DBSCAN + KNN)')
print(
    'Number of clusters: {}'
    .format(len(np.unique(data['Cluster_Hybrid']))))
print(
    'Silhouette score: {}'
    .format(round(silhouette_score(X,data['Cluster_Hybrid']),2)))