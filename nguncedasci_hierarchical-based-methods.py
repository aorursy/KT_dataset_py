from warnings import filterwarnings
filterwarnings('ignore')
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.cluster import KMeans
df=pd.read_csv("../input/usarrests/USArrests.csv").copy()
df.index = df.iloc[:,0]
df = df.iloc[:,1:5]
df.index.name = None
df.head()
from scipy.cluster.hierarchy import linkage
hc_complete = linkage(df, "complete")
hc_average = linkage(df, "average")
hc_single = linkage(df, "single")
from scipy.cluster.hierarchy import dendrogram

plt.figure(figsize=(15, 10))
plt.title('Hierarchical clustering - Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
dendrogram(
    hc_complete,
    leaf_font_size=10
);
from scipy.cluster.hierarchy import dendrogram

plt.figure(figsize=(15, 10))
plt.title('HHierarchical clustering - Dendrogram')
plt.xlabel('Cardinality')
plt.ylabel('Distance')
dendrogram(
    hc_complete,
    truncate_mode = "lastp",
    p = 4,                       #for 4 clusters
    show_contracted = True
);
df.shape
from sklearn.cluster import AgglomerativeClustering
Cluster=AgglomerativeClustering(n_clusters=4, affinity="euclidean",linkage="ward")   #We indicated that 4 is more preferable
Cluster.fit_predict(df)
pd.DataFrame({"States" : df.index, "Clusters": Cluster.fit_predict(df)})[0:10]
df["Clusters' Number"]=Cluster.fit_predict(df)
df.head()
