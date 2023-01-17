# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Plotting library
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#Read the training set and peek it
df = pd.read_csv('../input/data.csv')
df.head()
# The list of columns
df.columns
# Datatype of columns
df.dtypes
# More on the columns
df.info()
df.describe() ## Numerical
df.describe(include=['O']) # Objects
# Check for duplication 
df.iloc[:, 1:].duplicated().sum() 
# Few initialization
sns.set_style('whitegrid')
# Mapping Benign to 0 and Malignant to 1 and storing it in a different dataframe
y_test = pd.DataFrame()
y_test['diagnosis'] = df['diagnosis'].map({'M':0,'B':1}) # map({'M':1,'B':0})
df = df.drop('diagnosis',axis=1)
# This is done for ease of use when comparing with the results obtained with different methods for clustering
# Cleaning and modifying the data
df = df.drop('id',axis=1)
df = df.drop('Unnamed: 32',axis=1)
# Scaling the dataset
X = pd.DataFrame(preprocessing.scale(df.iloc[:,:]))
from scipy.cluster.hierarchy import ward, dendrogram, linkage
np.set_printoptions(precision=4, suppress=True)

distance = linkage(X, 'ward')
plt.figure(figsize=(25,10))
plt.title("Hierachical Clustering Dendrogram")
plt.xlabel("Index")
plt.ylabel("Ward's distance")
dendrogram(distance,
           leaf_rotation=90.,
           leaf_font_size=9.,);
plt.figure(figsize=(25,10))
plt.title("Hierachical Clustering Dendrogram")
plt.xlabel("Index")
plt.ylabel("Ward's distance")
dendrogram(distance,
           leaf_rotation=90.,
           leaf_font_size=9.,);
plt.axhline(60, c='k');
plt.axhline(100, c='k');
#By distance
from scipy.cluster.hierarchy import fcluster
max_d = 60
clusters = pd.DataFrame(fcluster(distance, max_d, criterion='distance'))
clusters[0] = clusters[0].map({1:1,2:0}) 
print(confusion_matrix(y_test, clusters))  
# By Number of Clusters         
k = 2
clusters = pd.DataFrame(fcluster(distance, k, criterion='maxclust'))  
clusters[0] = clusters[0].map({1:1,2:0}) 
print(confusion_matrix(y_test, clusters))         
clusters
# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

print(accuracy_score(y_test, y_hc))
print(classification_report(y_test, y_hc))
print(confusion_matrix(y_test, y_hc))
from sklearn.cluster import KMeans        
kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)   
 
print(accuracy_score(y_test, y_kmeans))
print(classification_report(y_test, y_kmeans))
print(confusion_matrix(y_test, y_kmeans))
# 1. Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
# 2. Silhouette Analysis
from sklearn.metrics import silhouette_score

sse_ = []
for k in range(2, 8):
    kmeans = KMeans(n_clusters=k).fit(X)
    sse_.append([k, silhouette_score(X, kmeans.labels_)])

plt.plot(pd.DataFrame(sse_)[0], pd.DataFrame(sse_)[1])
plt.title('Silhouette Analysis')
plt.xlabel('Number of clusters')
plt.ylabel('silhouette_score')
plt.show();
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle
bandwidth_X = estimate_bandwidth(X, quantile=0.1, n_samples=len(X))
meanshift_model = MeanShift(bandwidth=bandwidth_X, bin_seeding=True)
meanshift_model.fit(X)
meanshift_model.cluster_centers_
y_ms = meanshift_model.labels_
num_clusters = len(np.unique(y_ms))
print('\nNumber of clusters in input data =', num_clusters)
y_ms = pd.DataFrame(y_ms)
y_ms[0] = y_ms[0].map({1:0,0:1}) 
print(accuracy_score(y_test, y_ms))
print(classification_report(y_test, y_ms))
print(confusion_matrix(y_test,y_ms))
'Finished running all cluster algos'