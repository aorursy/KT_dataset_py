%matplotlib inline
import pandas as pd
import numpy as np
# importing ploting libraries

import matplotlib.pyplot as plt 
from scipy.stats import zscore
import seaborn as sns
import os

os.listdir('../input/wine-quality-clustering-unsupervised')
# reading the CSV file into pandas dataframe

wine_data = pd.read_csv("../input/wine-quality-clustering-unsupervised/winequality-red.csv") 
wine_data_attr = wine_data.iloc[:,0:12]

wine_data_attr.head()
features = ['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide', 'density','pH', 'sulphates','alcohol','quality']

#importing seaborn for statistical plots

import seaborn as sns





sns.pairplot(wine_data, size=7,aspect=0.5 , diag_kind='kde')
from sklearn.cluster import AgglomerativeClustering 
model = AgglomerativeClustering(n_clusters=6, affinity='euclidean',  linkage='average')
model.fit(wine_data_attr)
wine_data_attr['labels'] = model.labels_



wine_data_attr.groupby(["labels"]).count()
wine_clusters = wine_data_attr.groupby(['labels'])
wine_clusters
wine_groups=wine_clusters.head(1599)#This creates a pandas dataframegroupby object
from scipy.cluster.hierarchy import cophenet, dendrogram, linkage
from scipy.spatial.distance import pdist  #Pairwise distribution between data points
# cophenet index is a measure of the correlation between the distance of points in feature space and distance on dendrogram

# closer it is to 1, the better is the clustering



Z = linkage(wine_data_attr, 'average')

c, coph_dists = cophenet(Z , pdist(wine_data_attr))



c
plt.figure(figsize=(10, 10))

plt.title('Agglomerative Hierarchical Clustering Dendogram')

plt.xlabel('sample index')

plt.ylabel('Distance')

dendrogram(Z, leaf_rotation=90.,color_threshold = 40, leaf_font_size=8. )

plt.tight_layout()
# cophenet index is a measure of the correlation between the distance of points in feature space and distance on dendrogram

# closer it is to 1, the better is the clustering



Z = linkage(wine_data_attr, 'complete')

c, coph_dists = cophenet(Z , pdist(wine_data_attr))



c
plt.figure(figsize=(15, 15))

plt.title('Agglomerative Hierarchical Clustering Dendogram')

plt.xlabel('sample index')

plt.ylabel('Distance')

dendrogram(Z, leaf_rotation=90.,color_threshold=90,  leaf_font_size=10. )

plt.tight_layout()
# cophenet index is a measure of the correlation between the distance of points in feature space and distance on dendrogram

# closer it is to 1, the better is the clustering



Z = linkage(wine_data_attr, 'ward')

c, coph_dists = cophenet(Z , pdist(wine_data_attr))



c
plt.figure(figsize=(15, 15))

plt.title('Agglomerative Hierarchical Clustering Dendogram')

plt.xlabel('sample index')

plt.ylabel('Distance')

dendrogram(Z, leaf_rotation=90.,color_threshold=600,  leaf_font_size=10. )

plt.tight_layout()