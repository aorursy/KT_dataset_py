!pip install hdbscan
import matplotlib
%matplotlib inline
%config InlineBackend.figure_format = 'svg'
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import pandas as pd
import numpy as np

from tqdm import tqdm

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier

from ipywidgets import interactive

from collections import defaultdict

import hdbscan
import folium
import re


cols = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
        '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', 
        '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', 
        '#000075', '#808080']*10
data=pd.read_csv("../input/taxi-data/taxi_data.csv")
data.head()
data.duplicated(subset=['LON','LAT']).values.any()

data.isnull().values.any()
print(f'Before dropping NaNs and dupes\t:\tdf.shape = {data.shape}')
data.dropna(inplace=True)
data.drop_duplicates(subset=['LON','LAT'],keep='first',inplace=True)
print(f'After dropping NaNs and dupes\t:\tdf.shape = {data.shape}')
X=np.array(data[['LON','LAT']],dtype='float64')
plt.scatter(X[:,0],X[:,1],alpha=0.2,s=50)

m=folium.Map(location=[data.LAT.mean(),data.LON.mean()],zoom_start=10,
             tiles='OpenStreetMap')
for _,row in data.iterrows():
    folium.CircleMarker(
        location=[row.LAT,row.LON],
        radius=5,
        popup=re.sub(r'[^a-zA-Z ]+','',row.NAME),
        color='blue',
        fill=True
    ).add_to(m)
m
X_blobs, _=make_blobs(n_samples=1000,centers=10,n_features=2,
                      cluster_std=0.5,random_state=4)
plt.scatter(X_blobs[:,0],X_blobs[:,1],alpha=0.2)
# pred=np.load('Data/sample_clusters.npy')
# clu=np.unique(pred)
# for c in clu:
#     X=X_blobs[pred==c]
#     plt.scatter(X[:,0],X[:,1],alpha=0.2,c=cols[c])
# silhouette_score(X_blobs,pred)
# pred=np.load('Data/sample_clusters_improved.npy')
# clu=np.unique(pred)
# for c in clu:
#     X=X_blobs[pred==c]
#     plt.scatter(X[:,0],X[:,1],alpha=0.2,c=cols[c])
# silhouette_score(X_blobs,pred)
# X_blobs, _ = make_blobs(n_samples=1000, centers=50, 
#                         n_features=2, cluster_std=1, random_state=4)
# data = defaultdict(dict)
# for x in range(1,21):
#     model = KMeans(n_clusters=3, random_state=17, 
#                    max_iter=x, n_init=1).fit(X_blobs)
    
#     data[x]['class_predictions'] = model.predict(X_blobs)
#     data[x]['centroids'] = model.cluster_centers_
#     data[x]['unique_classes'] = np.unique(class_predictions)
# def f(x):
#     class_predictions = data[x]['class_predictions']
#     centroids = data[x]['centroids']
#     unique_classes = data[x]['unique_classes']

#     for unique_class in unique_classes:
#             plt.scatter(X_blobs[class_predictions==unique_class][:,0], 
#                         X_blobs[class_predictions==unique_class][:,1], 
#                         alpha=0.3, c=cols[unique_class])
#     plt.scatter(centroids[:,0], centroids[:,1], s=200, c='#000000', marker='v')
#     plt.ylim([-15,15]); plt.xlim([-15,15])
#     plt.title('How K-Means Clusters')

# interactive_plot = interactive(f, x=(1, 20))
# output = interactive_plot.children[-1]
# output.layout.height = '350px'
# interactive_plot
X=np.array(data[['LON','LAT']],dtype='float64')
k=70
model=KMeans(n_clusters=k,random_state=17).fit(X)
pred_k=model.predict(X)
data[f'CLUSTER_kmeans{k}']=pred_k
def create_map(data,cluster_col):
    m = folium.Map(location=[data.LAT.mean(), data.LON.mean()], zoom_start=9, tiles='openstreetmap')

    for _, row in data.iterrows():

        # get a colour
        if row[cluster_col]==-1:
            cluster_colour='black'
        else:
            cluster_colour = cols[row[cluster_col]]

        folium.CircleMarker(
            location=[row.LAT,row.LON],
            radius=5,
            popup= row[cluster_col],
            color=cluster_colour,
            fill=True,
            fill_color=cluster_colour
        ).add_to(m)
    return m

m=create_map(data,'CLUSTER_kmeans70')   
print(f'K={k}')
print(f'Silhouette Score: {silhouette_score(X, pred_k)}')

m.save('kmeans_70.html')
m
best_silhouette, best_k = -1, 0

for k in tqdm(range(2, 100)):
    model = KMeans(n_clusters=k, random_state=1).fit(X)
    class_predictions = model.predict(X)
    
    curr_silhouette = silhouette_score(X, class_predictions)
    if curr_silhouette > best_silhouette:
        best_k = k
        best_silhouette = curr_silhouette
        
print(f'K={best_k}')
print(f'Silhouette Score: {best_silhouette}') 
# code for indexing out certain values
dummy = np.array([-1, -1, -1, 2, 3, 4, 5, -1])
new=np.array([(counter+2)*x if x==-1 else x for counter,x in enumerate(dummy)])
new
model=DBSCAN(eps=0.01,min_samples=5).fit(X)
class_predictions=model.labels_
data['Clusters_dbscan']=class_predictions
m=create_map(data,'Clusters_dbscan')
print(f'Number of clusters found: {len(np.unique(class_predictions))}')
print(f'Number of outliers found: {len(class_predictions[class_predictions==-1])}')

print(f'Silhouette ignoring outliers: {silhouette_score(X[class_predictions!=-1], class_predictions[class_predictions!=-1])}')

no_outliers = 0
no_outliers = np.array([(counter+2)*x if x==-1 else x for counter, x in enumerate(class_predictions)])
print(f'Silhouette outliers as singletons: {silhouette_score(X, no_outliers)}')
m
model=hdbscan.HDBSCAN(min_cluster_size=5,min_samples=2,cluster_selection_epsilon=0.01)
#min_cluster_size
#min_samples
#cluster_selection_epsilon
class_predictions=model.fit_predict(X)
data['CLUSTER_hdbscan']=class_predictions
m=create_map(data,'CLUSTER_hdbscan')
print(f'Number of clusters found: {len(np.unique(class_predictions))-1}')
print(f'Number of outliers found: {len(class_predictions[class_predictions==-1])}')

print(f'Silhouette ignoring outliers: {silhouette_score(X[class_predictions!=-1], class_predictions[class_predictions!=-1])}')

no_outliers = np.array([(counter+2)*x if x==-1 else x for counter, x in enumerate(class_predictions)])
print(f'Silhouette outliers as singletons: {silhouette_score(X, no_outliers)}')

m

classifier=KNeighborsClassifier(n_neighbors=1)
data_train=data[data.CLUSTER_hdbscan!=-1]
data_test=data[data.CLUSTER_hdbscan==-1]
X_train=np.array(data_train[['LON','LAT']],dtype='float64')
y_train=np.array(data_train['CLUSTER_hdbscan'])

X_test=np.array(data_test[['LON',"LAT"]],dtype='float64')
classifier.fit(X_train,y_train)
pred=classifier.predict(X_test)
data['CLUSTER_hybrid']=data['CLUSTER_hdbscan']
data.loc[data.CLUSTER_hdbscan==-1,'CLUSTER_hybrid']=pred
m=create_map(data,'CLUSTER_hybrid')
m

class_predictions=data.CLUSTER_hybrid
print(f'Number of clusters found: {len(np.unique(class_predictions))}')
print(f'Silhouette: {silhouette_score(X, class_predictions)}')

m.save('hybrid.html')
data['CLUSTER_hybrid'].value_counts().plot.hist(bins=70,alpha=0.5,label='hybrid')
data['CLUSTER_kmeans70'].value_counts().plot.hist(bins=70,alpha=0.5,label='kmeans')
plt.legend()
plt.title('Comparing Hybrid and K-Means Approaches')
plt.xlabel('cluster sizes')

