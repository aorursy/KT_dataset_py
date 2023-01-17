import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#class1
x1 = np.random.normal(25,5,1000) 
y1 = np.random.normal(25,5,1000)
#class2
x2 = np.random.normal(55,5,1000) 
y2 = np.random.normal(60,5,1000)
#class3
x3 = np.random.normal(55,5,1000) 
y3 = np.random.normal(15,5,1000)

x = np.concatenate((x1,x2,x3),axis=0) #x1,x2,x3 birleştir, data sample 3000
y = np.concatenate((y1,y2,y3),axis=0) #y1,y2,y3 birleştir, data sample 3000

dictionary = {"x":x,"y":y}
data1 = pd.DataFrame(dictionary)

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,6))
ax1.scatter(x1,y1,color="black")
ax1.scatter(x2,y2,color="black")
ax1.scatter(x3,y3,color="black")
ax2.scatter(x1,y1)
ax2.scatter(x2,y2)
ax2.scatter(x3,y3)
fig.suptitle("Before and Target of Clustering")
ax1.set_title('None Clustering')
ax2.set_title('Target Clustering')
plt.show()
from sklearn.cluster import KMeans
wcss = []
wcss_values = range(1,15)

for k  in wcss_values:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data1)
    wcss.append(kmeans.inertia_) # Her K değeri için wcss değeri

plt.figure(figsize=[15,6])
plt.plot(wcss_values,wcss)
plt.xlabel("number of k (cluster) value")
plt.xticks(wcss_values)
plt.ylabel("wcss")
plt.show()
kmeans2 = KMeans(n_clusters=3)
clusters = kmeans2.fit_predict(data1) # fit et ve ardından predict et yani data üzerinde clusterları uygula.

data1["label"] = clusters
plt.figure(figsize=[15,6])
plt.title("After Clustering and last cluster centers")
plt.scatter(data1.x[data1.label==0],data1.y[data1.label==0],color="blue")
plt.scatter(data1.x[data1.label==1],data1.y[data1.label==1],color="red")
plt.scatter(data1.x[data1.label==2],data1.y[data1.label==2],color="green")
plt.scatter(kmeans2.cluster_centers_[:,0],kmeans2.cluster_centers_[:,1],color="black") #cluster centers
plt.show()
#class1
x1 = np.random.normal(25,5,100) 
y1 = np.random.normal(25,5,100)
#class2
x2 = np.random.normal(55,5,100) 
y2 = np.random.normal(60,5,100)
#class3
x3 = np.random.normal(55,5,100) 
y3 = np.random.normal(15,5,100)

x = np.concatenate((x1,x2,x3),axis=0) #x1,x2,x3 birleştir, data sample 3000
y = np.concatenate((y1,y2,y3),axis=0) #y1,y2,y3 birleştir, data sample 3000

dictionary = {"x":x,"y":y}
data1 = pd.DataFrame(dictionary)

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,6))
ax1.scatter(x1,y1,color="black")
ax1.scatter(x2,y2,color="black")
ax1.scatter(x3,y3,color="black")
ax2.scatter(x1,y1)
ax2.scatter(x2,y2)
ax2.scatter(x3,y3)
fig.suptitle("Before and Target of Clustering")
ax1.set_title('None Clustering')
ax2.set_title('Target Clustering')
plt.show()
#%% Dendogram
from scipy.cluster.hierarchy import linkage, dendrogram
#cluster içideki yayılımları,varyansları şekillendirebileceğimiz, minimalimize edeceğmiz yöntem.
plt.figure(figsize=(15,5))
merg = linkage(data1, method="ward") 
dendrogram(merg, leaf_rotation=90,orientation='top',truncate_mode='lastp',show_contracted=True)
plt.xlabel("data points")
plt.ylabel("euclidean distance")
plt.show()
from sklearn.cluster import AgglomerativeClustering

hierartical_cluster = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="ward")
clusters = hierartical_cluster.fit_predict(data1) #modeli hem oluştur hemde predict edip clusterları oluştur.

data1["label"] = clusters
plt.figure(figsize=(15,6))
plt.title('After Clustering')
plt.xlabel("x feature")
plt.ylabel("y feature")
plt.scatter(data1.x[data1.label==0],data1.y[data1.label==0],color="red")
plt.scatter(data1.x[data1.label==1],data1.y[data1.label==1],color="green")
plt.scatter(data1.x[data1.label==2],data1.y[data1.label==2],color="blue")
plt.show()
df = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
df.info()

df.Class.value_counts()
# 1 = sahtekarlık durumu (Fraud)
# 2 = Aksi durum         (OtherWise)
df.describe()
data = df.drop(['Class','Time'],axis=1)
data.head()
data.isnull().all()
#standartlaştırma kodu gelecek
data_loc = data.loc[:400:,['V11','V15']]
scaler_data = (data_loc - np.mean(data_loc)) / np.std(data_loc)
data2 = scaler_data
data2.head()
plt.scatter(data2.V11,data2.V15)
plt.xlabel('V11')
plt.ylabel('V15')
plt.show()

from sklearn.cluster import KMeans
wcss = []

for k  in range(1,15):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data2)
    wcss.append(kmeans.inertia_) # wcss value for each k value
plt.figure(figsize=(15,6))
plt.plot(range(1,15),wcss)
plt.xlabel("number of k (cluster) value")
plt.ylabel("wcss")
plt.show()
kmeans2 = KMeans(n_clusters = 5)
clusters = kmeans2.fit_predict(data2)
data2['label'] = clusters
plt.figure(figsize=[10,4])
plt.scatter(data2.V11,data2.V15,c = data2.label)
plt.scatter(kmeans2.cluster_centers_[:,0],kmeans2.cluster_centers_[:,1],color="red",linewidths=5)
plt.xlabel('V11')
plt.ylabel('V15')
plt.show()
data3 = data.loc[:200:,['V11','V15']]
plt.figure(figsize=[10,4])
plt.scatter(data2.V11,data2.V15,color='black')
plt.title("Before Clustering")
plt.xlabel('V11')
plt.ylabel('V15')
plt.show()


merge_single = linkage(data3, method='single',metric='euclidean' )
merge_complete = linkage(data3, method='complete',metric='euclidean')
merge_average = linkage(data3, method='average',metric='euclidean' )
merge_ward = linkage(data3, method='ward',metric='euclidean' )

fig, axes = plt.subplots(4, 1, figsize=(12, 24))

d_single=dendrogram(merge_single,ax=axes[0] ,leaf_rotation=90,truncate_mode='lastp',show_contracted=True)
d_complete=dendrogram(merge_complete, ax=axes[1],leaf_rotation=90,truncate_mode='lastp',show_contracted=True)
d_average=dendrogram(merge_average, ax=axes[2],leaf_rotation=90,truncate_mode='lastp',show_contracted=True)
d_ward=dendrogram(merge_ward, ax=axes[3],leaf_rotation=90,truncate_mode='lastp',show_contracted=True)

axes[0].set_title('Single')
axes[1].set_title('Complete')
axes[2].set_title('Average')
axes[3].set_title('Ward')
plt.show()

from sklearn.cluster import AgglomerativeClustering

hierartical_cluster = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="complete")
clusters = hierartical_cluster.fit_predict(data3) #modeli hem oluştur hemde predict edip clusterları oluştur.

plt.figure(figsize=(15,6))
data3["label"] = clusters
plt.scatter(data3.V11,data3.V15,c = data3.label)
plt.show()