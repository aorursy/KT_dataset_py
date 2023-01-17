# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer, MaxAbsScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, NMF
from mpl_toolkits.mplot3d import Axes3D
wheats = pd.read_csv("/kaggle/input/seed-from-uci/Seed_Data.csv")
wheats = wheats.rename(columns={"target":"label"})
wheats.head()
#Data Visualization - Label - countplot
fig,ax =  plt.subplots(figsize = (15 , 5))
sns.countplot(y = 'label' , data = wheats)
plt.show()
#Data Visualozation - Label vs other continuous variables - box plot
fig, axs = plt.subplots(ncols = 4, nrows = 2, figsize=(15,8),sharex=True)
for axi,col_name in zip(axs.flat,list(wheats.columns)):
    sns.boxplot(x="label",y=col_name,data=wheats,ax=axi)
fig.delaxes(ax = axs[1,3])
fig.tight_layout()
fig.show()
#Data Visualozation - Label vs other continuous variables - swarmplot
fig, axs = plt.subplots(ncols = 4, nrows = 2, figsize=(15,8),sharex=True)
for axi,col_name in zip(axs.flat,list(wheats.columns)):
    sns.swarmplot(x="label",y=col_name,data=wheats,ax=axi)
fig.delaxes(ax = axs[1,3])
fig.tight_layout()
fig.show()
#Data Visualization - Continuous Variables - pairplot
sns.pairplot(data=wheats,hue="label")
#Data Visualization - Continuous Variables - heatmap (Area, Compactness, Assymetry Coefficient)
sns.heatmap(wheats[list(wheats.columns[:-1])].corr(),annot=True,cmap="coolwarm")
plt.show()
#KMeans Clustering - Optimal number of clusters
inertia = []
for c in range(1,7):
    model = KMeans(n_clusters=c)
    model.fit(wheats[list(wheats.columns[:-1])])
    inertia.append(model.inertia_)
fig, ax = plt.subplots(figsize = (10 ,4))
ax.plot(np.arange(1 , 7) , inertia , marker="o")
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Inertia')
fig.show()
#KMeans Clustering - Validation
model = KMeans(n_clusters = 3)
model.fit(wheats[list(wheats.columns[:-1])])
label_pred = model.predict(wheats[list(wheats.columns[:-1])])
wheat_preds = pd.DataFrame({"label_act":list(wheats["label"]),"label_kmeans":list(label_pred)})
pd.crosstab(wheat_preds.label_act,wheat_preds.label_kmeans)
#k-Means Clustering - Standardisation
scaler = StandardScaler()
kmeans = KMeans(n_clusters=3)
pipeline = make_pipeline(scaler,kmeans)
pipeline.fit(wheats[list(wheats.columns[:-1])])
labels_pred_std = pipeline.predict(wheats[list(wheats.columns[:-1])])
wheat_preds["labels_kmeans_std"] = labels_pred_std
pd.crosstab(wheat_preds.label_act,wheat_preds.labels_kmeans_std)
samples = np.array(wheats[list(wheats.columns[:-1])])
samples.shape
#Agglomerative Clustering - Fathest point distance - Dendrogram
fig, ax = plt.subplots(figsize=(18,8))
mergings = linkage(samples,method="complete")
dendrogram(mergings,
           labels=list(wheats["label"]),
           leaf_rotation=90,
           leaf_font_size=6,
)
ax.plot(ax.get_xbound(), [8,8],'--', c='k')
fig.show()
#Agglomerative Clustering - Farthest point distance - Clustering
agg_clus = AgglomerativeClustering(n_clusters = 3,linkage="complete")
wheat_preds["labels_agg"] = agg_clus.fit_predict(wheats[list(wheats.columns[:-1])])
pd.crosstab(wheat_preds.label_act,wheat_preds.labels_agg)
#Agglomerative Clustering - Fathest point distance - Clustering - Standardisation
std = StandardScaler()
agg_clus = AgglomerativeClustering(n_clusters = 3,linkage="complete")
pipeline = make_pipeline(std,agg_clus)
pipeline.fit(wheats[list(wheats.columns[:-1])])
wheat_preds["labels_agg_std"] = pipeline.fit_predict(wheats[list(wheats.columns[:-1])])
pd.crosstab(wheat_preds.label_act,wheat_preds.labels_agg_std)
#Agglomerative Clustering - Fathest point distance - Clustering - Standardisation
std = StandardScaler()
agg_clus = AgglomerativeClustering(n_clusters = 3,linkage="ward",affinity="euclidean")
pipeline = make_pipeline(std,agg_clus)
pipeline.fit(wheats[list(wheats.columns[:-1])])
wheat_preds["labels_agg_std_ward"] = pipeline.fit_predict(wheats[list(wheats.columns[:-1])])
pd.crosstab(wheat_preds.label_act,wheat_preds.labels_agg_std_ward)
#t-SNE visualization - KMeans post standardisation
model = TSNE(learning_rate = 200)
tsne_features = model.fit_transform(samples)
sns.scatterplot(tsne_features[:,0],tsne_features[:,1],hue=list(wheat_preds.labels_kmeans_std),palette="Set2",s=100)
plt.legend(loc="lower right")
plt.show()
#t-SNE visualization - Agglomerative Clustering post standardisation
model = TSNE(learning_rate = 200)
tsne_features = model.fit_transform(samples)
sns.scatterplot(tsne_features[:,0],tsne_features[:,1],hue=list(wheat_preds.labels_agg_std_ward),palette="Set2",s=100)
plt.legend(loc="lower right")
plt.show()
#PCA - Explained Variance
scaler = StandardScaler()
pca = PCA()
pipeline = make_pipeline(scaler,pca)
pipeline.fit(wheats[list(wheats.columns[:-1])])
features = range(pca.n_components_)
sns.barplot(x=np.arange(0,7), y=pca.explained_variance_,palette="Blues")
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()
#PCA - Two components - Agglomerative Clustering post standardisation
pca_features = pipeline.transform(wheats[list(wheats.columns[:-1])])
sns.scatterplot(pca_features[:,0],pca_features[:,1],hue=list(wheat_preds.labels_agg_std_ward),palette="copper",s=100)
plt.show()
#PCA - Three components - K Means Clustering post standardisation
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_features[:,0],pca_features[:,1],pca_features[:,2],c=list(wheat_preds.labels_kmeans_std),alpha=0.6)
fig.show()
#NMF - Interpretability
model = NMF(n_components = 2,max_iter=1000)
nmf_features = model.fit_transform(wheats[list(wheats.columns[:-1])])
components_df = pd.DataFrame(model.components_,columns=list(wheats.columns[:-1]))
components_df
#NMF - Visualization
nmf = NMF(n_components = 2,max_iter=1000)
norm_features = nmf.fit_transform(wheats[list(wheats.columns[:-1])])
sns.scatterplot(norm_features[:,0],norm_features[:,1],hue=list(wheat_preds.labels_agg_std_ward),palette="Set2",alpha=0.6,s=100)
plt.show()
