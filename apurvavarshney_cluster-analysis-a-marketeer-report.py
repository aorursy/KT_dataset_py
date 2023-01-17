import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from numpy import unique
from numpy import where
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
%matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv("/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")

df.head()
df.drop(["CustomerID"], axis = 1, inplace = True)
df.info()
encoder = LabelEncoder()
Gender_ec = encoder.fit_transform(df.iloc[:,0])
df["Gender"] = Gender_ec
df.head()
scaler = StandardScaler()
scaled = scaler.fit_transform(df)
df1 = pd.DataFrame(data = scaled, columns = ["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"])
df1.head()
pca = PCA(n_components = 2)
df2 = pca.fit_transform(df1)
df2.shape
plt.scatter(df2[:, 0], df2[:, 1])
from sklearn.cluster import KMeans

model = KMeans(n_clusters = 5)
yhat = model.fit_predict(df2)
clusters = unique(yhat)
for cluster in clusters:
    row_ix = where(yhat == cluster)
    plt.scatter(df2[row_ix, 0], df2[row_ix, 1])
    plt.title("Sklearn version of KMeans cluster")
    plt.style.use('fivethirtyeight')
print(model.labels_)
# Nice Pythonic way to get the indices of the points for each corresponding cluster
mydict = {i: np.where(model.labels_ == i)[0] for i in range(model.n_clusters)}

# Transform this dictionary into list (if you need a list as result)
dictlist = []
for key, value in mydict.items():
    temp = [key, value]
    dictlist.append(temp)
#This list contains indices of objects in the cluster
dictlist[0]
## To get the array of our original encoded dataset 
df3 = df.values
## To get items from the original dataset
accessed_mapping = map(df3.__getitem__, dictlist[0])
cl1 = list(accessed_mapping)
cluster_1 = pd.DataFrame(cl1[1], columns = ["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"])

accessed_mapping = map(df3.__getitem__, dictlist[1])
cl2 = list(accessed_mapping)
cluster_2 = pd.DataFrame(cl2[1], columns = ["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"])

accessed_mapping = map(df3.__getitem__, dictlist[2])
cl3 = list(accessed_mapping)
cluster_3 = pd.DataFrame(cl3[1], columns = ["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"])

accessed_mapping = map(df3.__getitem__, dictlist[3])
cl4 = list(accessed_mapping)
cluster_4 = pd.DataFrame(cl4[1], columns = ["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"])

accessed_mapping = map(df3.__getitem__, dictlist[4])
cl5 = list(accessed_mapping)
cluster_5 = pd.DataFrame(cl5[1], columns = ["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"])

## The objects in clusters
cluster_1
## Final Report of Cluster 1


print("*" * 75)
print("The Average age of Customers in cluster 1 is:")
print(cluster_1.Age.mean())
print("*" * 75)
print("The Number of Male(1) and female(0) customers in cluster 1 are:")
print(cluster_1["Gender"].value_counts())
print("*" * 75)
print("The Average annual income (in dollars) of Customers in category 1 is:")
print(cluster_1["Annual Income (k$)"].mean())
print("*" * 75)
print("The Mean,Median and Mode of spending Score of people in category 1 is:")
print(cluster_1["Spending Score (1-100)"].mode())
print("*" * 75)
## Final Report of Cluster 2

print("*" * 75)
print("The Average age of Customers in cluster 2 is:")
print(cluster_2.Age.mean())
print("*" * 75)
print("The Number of Male(1) and female(0) customers in cluster 2 are:")
print(cluster_2["Gender"].value_counts())
print("*" * 75)
print("The Average annual income (in dollars) of Customers in category 2 is:")
print(cluster_2["Annual Income (k$)"].mean())
print("*" * 75)
print("The Mean,Median and Mode of spending Score of people in category 2 is:")
print(cluster_2["Spending Score (1-100)"].mode())
print("*" * 75)
## Final Report of Cluster 3

print("*" * 75)
print("The Average age of Customers in cluster 3 is:")
print(cluster_3.Age.mean())
print("*" * 75)
print("The Number of Male(1) and female(0) customers in cluster 3 are:")
print(cluster_3["Gender"].value_counts())
print("*" * 75)
print("The Average annual income (in dollars) of Customers in category 3 is:")
print(cluster_3["Annual Income (k$)"].mean())
print("*" * 75)
print("The Mean,Median and Mode of spending Score of people in category 3 is:")
print(cluster_3["Spending Score (1-100)"].mode())
print("*" * 75)
## Final Report of Cluster 4

print("*" * 75)
print("The Average age of Customers in cluster 4 is:")
print(cluster_4.Age.mean())
print("*" * 75)
print("The Number of Male(1) and female(0) customers in cluster 4 are:")
print(cluster_4["Gender"].value_counts())
print("*" * 75)
print("The Average annual income (in dollars) of Customers in category 4 is:")
print(cluster_4["Annual Income (k$)"].mean())
print("*" * 75)
print("The Mean,Median and Mode of spending Score of people in category 4 is:")
print(cluster_4["Spending Score (1-100)"].mode())
print("*" * 75)
## Final Report of Cluster 5

print("*" * 75)
print("The Average age of Customers in cluster 5 is:")
print(cluster_5.Age.mean())
print("*" * 75)
print("The Number of Male(1) and female(0) customers in cluster 5 are:")
print(cluster_5["Gender"].value_counts())
print("*" * 75)
print("The Average annual income (in dollars) of Customers in category 5 is:")
print(cluster_5["Annual Income (k$)"].mean())
print("*" * 75)
print("The Mean,Median and Mode of spending Score of people in category 5 is:")
print(cluster_5["Spending Score (1-100)"].mode())
print("*" * 75)
from sklearn.cluster import AffinityPropagation
model = AffinityPropagation(damping=0.9)
model.fit(df2)
yhat = model.predict(df2)
clusters = unique(yhat)
for cluster in clusters:
    row_ix = where(yhat == cluster)
    plt.scatter(df2[row_ix, 0], df2[row_ix, 1])
    #plt.scatter(cluster.cluster_centers_[:, 0], cluster.cluster_centers_[:, 1], marker = '+', label='Clusters', c = "red")
    plt.title("Sklearn version of Affinity Propagation")
    plt.style.use('fivethirtyeight')
from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters = 5)
yhat = model.fit_predict(df2)
clusters = unique(yhat)
for cluster in clusters:
    row_ix = where(yhat == cluster)
    plt.scatter(df2[row_ix, 0], df2[row_ix, 1])
    plt.title("Sklearn version of Agglomerative Clustering")
    plt.style.use('fivethirtyeight')
    
from sklearn.cluster import Birch
model = Birch(threshold=0.01, n_clusters=5)
model.fit(df2)
yhat = model.predict(df2)
clusters = unique(yhat)
for cluster in clusters:
    row_ix = where(yhat == cluster)
    plt.scatter(df2[row_ix, 0], df2[row_ix, 1])
##plt.scatter(cluster.cluster_centers_[:, 0], cluster.cluster_centers_[:, 1], marker = '+', label='Clusters', c = "red")
    plt.title("Sklearn version of BIRCH")
    plt.style.use('fivethirtyeight')
from sklearn.cluster import DBSCAN
model = DBSCAN(eps=0.30)
yhat = model.fit_predict(df2)
clusters = unique(yhat)
for cluster in clusters:
    row_ix = where(yhat == cluster)
    plt.scatter(df2[row_ix, 0], df2[row_ix, 1])
    plt.title("Sklearn version of DBSCAN")
    plt.style.use('fivethirtyeight')
from sklearn.cluster import MiniBatchKMeans
model = MiniBatchKMeans(n_clusters=5)
model.fit(df2)
yhat = model.predict(df2)
clusters = unique(yhat)
for cluster in clusters:
    row_ix = where(yhat == cluster)
    plt.scatter(df2[row_ix, 0], df2[row_ix, 1])
##plt.scatter(cluster.cluster_centers_[:, 0], cluster.cluster_centers_[:, 1], marker = '+', label='Clusters', c = "red")
    plt.title("Sklearn version of Mini Batch Means")
    plt.style.use('fivethirtyeight')
from sklearn.cluster import OPTICS

model = OPTICS(eps=0.8)
yhat = model.fit_predict(df2)
clusters = unique(yhat)
for cluster in clusters:
    row_ix = where(yhat == cluster)
    plt.scatter(df2[row_ix, 0], df2[row_ix, 1])
    plt.title("Sklearn version of optics clustering")
    plt.style.use('fivethirtyeight')
from sklearn.cluster import SpectralClustering

model = SpectralClustering(n_clusters = 5)
yhat = model.fit_predict(df2)
clusters = unique(yhat)
for cluster in clusters:
    row_ix = where(yhat == cluster)
    plt.scatter(df2[row_ix, 0], df2[row_ix, 1])
    plt.title("Sklearn version of Spectral clustering")
    plt.style.use('fivethirtyeight')
from sklearn.mixture import GaussianMixture

model = GaussianMixture(n_components = 5)
yhat = model.fit_predict(df2)
clusters = unique(yhat)
for cluster in clusters:
    row_ix = where(yhat == cluster)
    plt.scatter(df2[row_ix, 0], df2[row_ix, 1])
    plt.title("Sklearn version of Gaussian Mixture")
    plt.style.use('fivethirtyeight')
