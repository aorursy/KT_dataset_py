import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.preprocessing import StandardScaler
from matplotlib import cm
from sklearn.metrics import silhouette_samples

import os
print(os.listdir("../input"))
df=pd.read_excel("../input/Data_bollywood.xlsx")
print(df.head())
print(df.info())
df.head()
df1=df.drop('Movie_Name',axis=1)
df1.head()
print(df1['Box_Office_Collection'].describe())
plt.figure(figsize=(7,6))
sns.distplot(df1['Box_Office_Collection'])
print(df1['Profit'].describe())
plt.figure(figsize=(7,6))
sns.distplot(df1['Profit'])
print(df1['Earning_Ratio'].describe())
plt.figure(figsize=(7,6))
sns.distplot(df1['Earning_Ratio'])
print(df1['Budget'].describe())
plt.figure(figsize=(7,6))
sns.distplot(df1['Budget'])
print(df1['Youtube_Views'].describe())
plt.figure(figsize=(7,6))
sns.distplot(df1['Youtube_Views'])
print(df1['Youtube_Likes'].describe())
plt.figure(figsize=(7,6))
sns.distplot(df1['Youtube_Likes'])
print(df1['Youtube_Dislikes'].describe())
plt.figure(figsize=(7,6))
sns.distplot(df1['Youtube_Dislikes'])
sns.pairplot(df1)
df1.hist(figsize=(7,6),bins=50,xlabelsize=8,ylabelsize=8)
scaler = StandardScaler()
scaler.fit(df.drop('Movie_Name',axis=1))
scaled_features = scaler.transform(df.drop('Movie_Name',axis=1))
print("Type of scaled_feature:",type(scaled_features))
print(scaled_features)
df2= pd.DataFrame(scaled_features,columns=df.columns[1:8])
df2.head()
X=df2.values
print(X)
plt.scatter(X[:,0],X[:,6])
plt.show()
km = KMeans(n_clusters=3, 
            init='random', 
            n_init=10, 
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)

plt.scatter(X[y_km==0,0], 
            X[y_km==0,1], 
            s=50, 
            c='lightgreen', 
            marker='s', 
            label='cluster 1')
plt.scatter(X[y_km==1,0], 
            X[y_km==1,1], 
            s=50, 
            c='orange', 
            marker='o', 
            label='cluster 2')
plt.scatter(X[y_km==2,0], 
            X[y_km==2,1], 
            s=50, 
            c='lightblue', 
            marker='v', 
            label='cluster 3')
plt.scatter(km.cluster_centers_[:,0], 
            km.cluster_centers_[:,1], 
            s=250, 
            marker='*', 
            c='red', 
            label='centroids')
plt.legend()
plt.grid()
plt.tight_layout()
#plt.savefig('./figures/centroids.png', dpi=300)
plt.show()
print('Distortion: %.2f' % km.inertia_)
distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, 
                init='k-means++', 
                n_init=10, 
                max_iter=300, 
                random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)
plt.plot(range(1,11), distortions , marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.tight_layout()
#plt.savefig('./figures/elbow.png', dpi=300)
plt.show()
import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_samples

km = KMeans(n_clusters=7, 
            init='k-means++', 
            n_init=10, 
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)

cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km==c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(i / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, 
            edgecolor='none', color=color)

    yticks.append((y_ax_lower + y_ax_upper) / 2)
    y_ax_lower += len(c_silhouette_vals)
    
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--") 

plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')

plt.tight_layout()
# plt.savefig('./figures/silhouette.png', dpi=300)
plt.show()
import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_samples

km = KMeans(n_clusters=6, 
            init='k-means++', 
            n_init=10, 
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)

cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km==c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(i / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, 
            edgecolor='none', color=color)

    yticks.append((y_ax_lower + y_ax_upper) / 2)
    y_ax_lower += len(c_silhouette_vals)
    
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--") 

plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')

plt.tight_layout()
# plt.savefig('./figures/silhouette.png', dpi=300)
plt.show()
import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_samples

km = KMeans(n_clusters=5, 
            init='k-means++', 
            n_init=10, 
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)

cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km==c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(i / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, 
            edgecolor='none', color=color)

    yticks.append((y_ax_lower + y_ax_upper) / 2)
    y_ax_lower += len(c_silhouette_vals)
    
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--") 

plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')

plt.tight_layout()
# plt.savefig('./figures/silhouette.png', dpi=300)
plt.show()
import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_samples

km = KMeans(n_clusters=4, 
            init='k-means++', 
            n_init=10, 
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)

cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km==c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(i / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, 
            edgecolor='none', color=color)

    yticks.append((y_ax_lower + y_ax_upper) / 2)
    y_ax_lower += len(c_silhouette_vals)
    
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--") 

plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')

plt.tight_layout()
# plt.savefig('./figures/silhouette.png', dpi=300)
plt.show()
import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_samples

km = KMeans(n_clusters=3, 
            init='k-means++', 
            n_init=10, 
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)

cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km==c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(i / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, 
            edgecolor='none', color=color)

    yticks.append((y_ax_lower + y_ax_upper) / 2)
    y_ax_lower += len(c_silhouette_vals)
    
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--") 

plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')

plt.tight_layout()
# plt.savefig('./figures/silhouette.png', dpi=300)
plt.show()
km = KMeans(n_clusters=5, 
            init='k-means++', 
            n_init=10, 
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)

cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km==c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(i / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, 
            edgecolor='none', color=color)

    yticks.append((y_ax_lower + y_ax_upper) / 2)
    y_ax_lower += len(c_silhouette_vals)
    
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--") 

plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')

plt.tight_layout()
# plt.savefig('./figures/silhouette.png', dpi=300)
plt.show()
a = km.cluster_centers_[0]
b = km.cluster_centers_[2]
d = km.cluster_centers_[4]
c = np.vstack((a,b,d))
c
feat = df2.as_matrix()

plt.scatter(feat[y_km==0,0], 
            feat[y_km==0,3], 
            s=50, 
            c='lightgreen', 
            marker='s', 
            label='cluster 1')
plt.scatter(feat[y_km==2,0], 
            feat[y_km==2,3], 
            s=50, 
            c='orange', 
            marker='o', 
            label='cluster 3')
plt.scatter(feat[y_km==4,0], 
            feat[y_km==4,3], 
            s=50, 
            c='blue', 
            marker='v', 
            label='cluster 5')
plt.scatter(c[:,0], 
            c[:,3], 
            s=250, 
            marker='*', 
            c='red', 
            label='centroids')
plt.legend()
plt.grid()
plt.tight_layout()
plt.xlabel('Box Office Collections')
plt.ylabel('Budget')
plt.suptitle('Box office collections v/s budget')
#plt.savefig('./figures/centroids.png', dpi=300)
plt.show()
plt.scatter(feat[y_km==0,0], 
            feat[y_km==0,4], 
            s=50, 
            c='lightgreen', 
            marker='s', 
            label='cluster 1')
plt.scatter(feat[y_km==2,0], 
            feat[y_km==2,4], 
            s=50, 
            c='orange', 
            marker='o', 
            label='cluster 3')
plt.scatter(feat[y_km==4,0], 
            feat[y_km==4,4], 
            s=50, 
            c='blue', 
            marker='v', 
            label='cluster 5')
plt.scatter(c[:,0], 
            c[:,4], 
            s=250, 
            marker='*', 
            c='red', 
            label='centroids')
plt.legend()
plt.grid()
plt.tight_layout()
plt.xlabel('Box Office Collections')
plt.ylabel('YouTube Views')
plt.suptitle('Box office collections v/s YouTube Views')
#plt.savefig('./figures/centroids.png', dpi=300)
plt.show()
plt.scatter(feat[y_km==0,1], 
            feat[y_km==0,6], 
            s=50, 
            c='lightgreen', 
            marker='s', 
            label='cluster 1')
plt.scatter(feat[y_km==2,1], 
            feat[y_km==2,6], 
            s=50, 
            c='orange', 
            marker='o', 
            label='cluster 3')
plt.scatter(feat[y_km==4,1], 
            feat[y_km==4,6], 
            s=50, 
            c='blue', 
            marker='v', 
            label='cluster 5')
plt.scatter(c[:,1], 
            c[:,6], 
            s=250, 
            marker='*', 
            c='red', 
            label='centroids')
plt.legend()
plt.grid()
plt.tight_layout()
plt.xlabel('Profit')
plt.ylabel('YouTube Dislikes')
plt.suptitle('Profit v/s YouTube Dislikes')
#plt.savefig('./figures/centroids.png', dpi=300)
plt.show()
plt.scatter(feat[y_km==0,1], 
            feat[y_km==0,2], 
            s=50, 
            c='lightgreen', 
            marker='s', 
            label='cluster 1')
plt.scatter(feat[y_km==2,1], 
            feat[y_km==2,2], 
            s=50, 
            c='orange', 
            marker='o', 
            label='cluster 3')
plt.scatter(feat[y_km==4,1], 
            feat[y_km==4,2], 
            s=50, 
            c='blue', 
            marker='v', 
            label='cluster 5')
plt.scatter(c[:,1], 
            c[:,2], 
            s=250, 
            marker='*', 
            c='red', 
            label='centroids')
plt.legend()
plt.grid()
plt.tight_layout()
plt.xlabel('Profit')
plt.ylabel('Earning Ratio')
plt.suptitle('Profit v/s Earning Ratio')
#plt.savefig('./figures/centroids.png', dpi=300)
plt.show()
plt.scatter(feat[y_km==0,4], 
            feat[y_km==0,5], 
            s=50, 
            c='lightgreen', 
            marker='s', 
            label='cluster 1')
plt.scatter(feat[y_km==2,4], 
            feat[y_km==2,5], 
            s=50, 
            c='orange', 
            marker='o', 
            label='cluster 3')
plt.scatter(feat[y_km==4,4], 
            feat[y_km==4,5], 
            s=50, 
            c='blue', 
            marker='v', 
            label='cluster 5')
plt.scatter(c[:,4], 
            c[:,5], 
            s=250, 
            marker='*', 
            c='red', 
            label='centroids')
plt.legend()
plt.grid()
plt.tight_layout()
plt.xlabel('Youtube_Views')
plt.ylabel('Youtube_Likes')
plt.suptitle('Youtube_Likes v/s Youtube_Views')
#plt.savefig('./figures/centroids.png', dpi=300)
plt.show()
feat = df2.as_matrix()

plt.scatter(feat[y_km==0,1], 
            feat[y_km==0,3], 
            s=50, 
            c='lightgreen', 
            marker='s', 
            label='cluster 1')
plt.scatter(feat[y_km==1,1], 
            feat[y_km==1,3], 
            s=50, 
            c='orange', 
            marker='o', 
            label='cluster 2')
plt.scatter(feat[y_km==4,1], 
            feat[y_km==4,3], 
            s=50, 
            c='blue', 
            marker='v', 
            label='cluster 5')
plt.scatter(c[:,1], 
            c[:,3], 
            s=250, 
            marker='*', 
            c='red', 
            label='centroids')
plt.legend()
plt.grid()
plt.tight_layout()
plt.xlabel('Profit')
plt.ylabel('Budget')
plt.suptitle('Profit v/s budget')
#plt.savefig('./figures/centroids.png', dpi=300)
plt.show()
f1 = df['Box_Office_Collection'].values
f2 = df['Profit'].values
f3 = df['Earning_Ratio'].values
f4 = df['Budget'].values
f5 = df['Youtube_Views'].values
f6 = df['Youtube_Likes'].values
f7 = df['Youtube_Dislikes'].values

X=np.matrix(list(zip(f1,f2,f3,f4,f5,f6,f7)))
km_1 = KMeans(n_clusters=5).fit(X)
labels = km_1.labels_
cluster_centers = km_1.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)
colors = cycle('bgrymckbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 4], col + '.', label=k+1)
    plt.plot(cluster_center[0], cluster_center[4], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.xlabel('Box Office Collections')
plt.ylabel('YouTube Views')
plt.legend()
plt.show()