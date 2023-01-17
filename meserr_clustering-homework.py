import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
#read csv file and get two columns in data
data = pd.read_csv("../input/column_2C_weka.csv")
data2 = data.loc[:,['degree_spondylolisthesis','pelvic_radius']]
#find optimum wcss value for cluster count

wcss = []
for k in range(1,15):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data2)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,15), wcss,'-o')
plt.xlabel("number of k (cluster) value")
plt.ylabel("wcss")
plt.show()
#create kmeans model and predict data

kmeans2 = KMeans(n_clusters=2)
clusters = kmeans2.fit_predict(data2)
data2['label'] = clusters
#Visualization cluster
plt.scatter(data['pelvic_radius'],data['degree_spondylolisthesis'],c = clusters)
plt.xlabel('pelvic_radius')
plt.xlabel('degree_spondylolisthesis')
plt.show()
# cross tabulation table
df = pd.DataFrame({'labels':data2['label'],"class":data['class']})
ct = pd.crosstab(df['labels'],df['class'])
print(ct)
# create data3 from data for hierarcical clustering
data3 = data.drop(['class'], axis=1)
data3 = data3.iloc[200:300,:]
#find optimum cluster count using dendogram

merg = linkage(data3,method="ward")
dendrogram(merg, leaf_rotation=90)
plt.xlabel("data points")
plt.ylabel("euclidean distance")
plt.show()
#create hierarcical clustering model

hierarchical_cluster = AgglomerativeClustering(n_clusters=2,affinity="euclidean", linkage="ward")
cluster = hierarchical_cluster.fit_predict(data3)
data3['label'] = cluster
# cross tabulation table
df = pd.DataFrame({'labels':data3['label'],"class":data['class']})
ct = pd.crosstab(df['labels'],df['class'])
print(ct)
#Visualization of find cluster by model
plt.scatter(data3.pelvic_radius[data3['label'] ==0], data3.degree_spondylolisthesis[data3.label==0],color="red")
plt.scatter(data3.pelvic_radius[data3['label'] ==1], data3.degree_spondylolisthesis[data3.label==1],color="blue")
plt.show()
