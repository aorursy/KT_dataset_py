import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import scipy.cluster.hierarchy as cls

# read file
#data.csv has some problems
data = pd.read_csv("/kaggle/input/testdata/data.csv", index_col=0, skiprows=1, header=None)
#use manually fixed csv file data1.csv
#data = pd.read_csv("/kaggle/input/testdata/data1.csv", index_col=0)
data.head()
data.shape
data.describe(include='all')
# check if there are missing data
data.isnull().any().any()
# Standardize the data
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
data_s = StandardScaler().fit_transform(data)
# Sanity check 
print("Summary of standardized data: ")
print("Mean", np.mean(data_s, axis=0))
print("Variance", np.var(data_s, axis=0))
print("Min", np.min(data_s, axis=0))
print("Max", np.max(data_s, axis=0))
#normalize: Scale input vectors individually to unit norm (vector length). 
#options for norm: l1, l2, 'max', default=l2
#default axis=1, across columns; need to set axis=0 to normalize individual features
data_n = normalize(data, norm='max', axis=0)
print("Summary of normalized data (norm: max)")
print("Mean", np.mean(data_n, axis=0))
print("Variance", np.var(data_n, axis=0))
print("Min", np.min(data_n, axis=0))
print("Max", np.max(data_n, axis=0))
# clustering (on original data)
plt.figure(figsize=(10, 7))  
plt.title("Dendrogram (orginal data)")  
#call linkage to prepare the linkage_matrix, and then call dendrogram to visualize
clustering = cls.linkage(data, method='complete', metric='euclidean')
# dendrogram
dend = cls.dendrogram(clustering)
# you may use the plot_dendrogram function as following to get the same visualization
# clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None, affinity='euclidean', linkage="complete").fit(data)
# plot_dendrogram(clustering, truncate_mode='none')
# clustering (on normalized data)
plt.figure(figsize=(10, 7))  
plt.title("Dendrogram (normalized data)")  
clustering = cls.linkage(data_n, method='complete', metric='euclidean')
# dendrogram
dend = cls.dendrogram(clustering)
# clustering (on standardized data)
plt.figure(figsize=(10, 7))  
plt.title("Dendrogram (standardized data)")  
clustering = cls.linkage(data_s, method='complete', metric='euclidean')
# dendrogram
dend = cls.dendrogram(clustering)