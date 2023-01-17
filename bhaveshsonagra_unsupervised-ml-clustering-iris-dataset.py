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

df = pd.read_csv("/kaggle/input/iris/Iris.csv")
df.head()

df.head()
### Drop Id column dont reqired this is Unwanted column
df.drop(["Id"],axis=1,inplace=True)   # dropped

df.head()
df.shape
df.info()
import matplotlib.pyplot as plt
import seaborn as sns
sns.pairplot(data=df,hue='Species',palette='pink')
df.columns
data=df.loc[:,['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=100)
#### Finding the best amount of clusters to get most accurate results (KMeans)
from sklearn.cluster import KMeans
wcss = [] # within cluster sum of square
for k in range(1,15):
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)
    
plt.figure(figsize=(20,8))
plt.title("The Elbow method", fontsize=18)
plt.plot(range(1,15),wcss,"-o")
plt.grid(True)
plt.xlabel("Amount of Clusters",fontsize=14)
plt.ylabel("Inertia",fontsize=14)
plt.xticks(range(1,20))
plt.tight_layout()
plt.show()
### Above the figure looks like elbow method
plt.figure(figsize=(24,4))

plt.suptitle("K Means Clustering",fontsize=20)

plt.subplot(1,5,1)
plt.title('k=1',fontsize=15)
plt.xlabel('PetalLengthCm')
plt.ylabel('PetalWidthCm')
plt.scatter(data.PetalLengthCm,data.PetalWidthCm)


plt.subplot(1,5,2)
plt.title('k=2',fontsize=15)
plt.xlabel('PetalLengthCm')
kmeans = KMeans(n_clusters=2)
data['Labels']=kmeans.fit_predict(data)
plt.scatter(data.PetalLengthCm[data.Labels == 0],data.PetalWidthCm[data.Labels == 0])
plt.scatter(data.PetalLengthCm[data.Labels == 1],data.PetalWidthCm[data.Labels == 1])


data.drop(['Labels'],axis=1,inplace=True)


plt.subplot(1,5,3)
plt.title('k=3',fontsize=15)
plt.xlabel('PetalLengthCm')
kmeans = KMeans(n_clusters=3)
data['Labels']=kmeans.fit_predict(data)
plt.scatter(data.PetalLengthCm[data.Labels == 0],data.PetalWidthCm[data.Labels == 0])
plt.scatter(data.PetalLengthCm[data.Labels == 1],data.PetalWidthCm[data.Labels == 1])
plt.scatter(data.PetalLengthCm[data.Labels == 2],data.PetalWidthCm[data.Labels == 2])

data.drop(["Labels"],axis=1,inplace=True)


plt.subplot(1,5,4)
plt.title('k=4',fontsize=15)
plt.xlabel('PetalLengthCm')
kmeans = KMeans(n_clusters=4)
data['Labels']=kmeans.fit_predict(data)
plt.scatter(data.PetalLengthCm[data.Labels == 0],data.PetalWidthCm[data.Labels == 0])
plt.scatter(data.PetalLengthCm[data.Labels == 1],data.PetalWidthCm[data.Labels == 1])
plt.scatter(data.PetalLengthCm[data.Labels == 2],data.PetalWidthCm[data.Labels == 2])
plt.scatter(data.PetalLengthCm[data.Labels == 3],data.PetalWidthCm[data.Labels == 3])


data.drop(["Labels"],axis=1,inplace=True)



plt.subplot(1,5,5)
plt.title('original labels',fontsize=15)
plt.xlabel('PetalLengthCm')
plt.scatter(df.PetalLengthCm[df.Species == 'Iris-setosa'],df.PetalWidthCm[df.Species == 'Iris-setosa'])
plt.scatter(df.PetalLengthCm[df.Species == 'Iris-virginica'],df.PetalWidthCm[df.Species == 'Iris-virginica'])
plt.scatter(df.PetalLengthCm[df.Species == 'Iris-versicolor'],df.PetalWidthCm[df.Species == 'Iris-versicolor'])


plt.subplots_adjust(top=.8)
plt.show()
#data['Labels']=kmeans.fit_predict(data)
#df['Species'].value_counts()
#data.head()
#data['Labels'].value_counts()
from sklearn.cluster import AgglomerativeClustering
hc_cluster = AgglomerativeClustering(n_clusters=100) 
## Finding the best amount of clusters to get most accurate results (Hierarchy)
from scipy.cluster.hierarchy import dendrogram,linkage
merge=linkage(data,method='ward')

plt.figure(figsize=(18,6))
dendrogram(merge,leaf_rotation=90)
plt.xlabel("data points")
plt.ylabel("euclidian distance")

plt.suptitle("DENDROGRAM",fontsize=18)
plt.show()
plt.figure(figsize=(24,4))

plt.suptitle("Hierarchical Clustering",fontsize=20)

plt.subplot(1,5,1)
plt.title('k=1',fontsize=15)
plt.xlabel('PetalLengthCm')
plt.ylabel('PetalWidthCm')
plt.scatter(data.PetalLengthCm,data.PetalWidthCm)


plt.subplot(1,5,2)
plt.title('k=2',fontsize=15)
plt.xlabel('PetalLengthCm')
hc_cluster = AgglomerativeClustering(n_clusters=2)
data['Labels']=kmeans.fit_predict(data)
plt.scatter(data.PetalLengthCm[data.Labels == 0],data.PetalWidthCm[data.Labels == 0])
plt.scatter(data.PetalLengthCm[data.Labels == 1],data.PetalWidthCm[data.Labels == 1])


data.drop(['Labels'],axis=1,inplace=True)


plt.subplot(1,5,3)
plt.title('k=3',fontsize=15)
plt.xlabel('PetalLengthCm')
hc_cluster = AgglomerativeClustering(n_clusters=3)
data['Labels']=kmeans.fit_predict(data)
plt.scatter(data.PetalLengthCm[data.Labels == 0],data.PetalWidthCm[data.Labels == 0])
plt.scatter(data.PetalLengthCm[data.Labels == 1],data.PetalWidthCm[data.Labels == 1])
plt.scatter(data.PetalLengthCm[data.Labels == 2],data.PetalWidthCm[data.Labels == 2])

data.drop(["Labels"],axis=1,inplace=True)


plt.subplot(1,5,4)
plt.title('k=4',fontsize=15)
plt.xlabel('PetalLengthCm')
hc_cluster = AgglomerativeClustering(n_clusters=4)
data['Labels']=kmeans.fit_predict(data)
plt.scatter(data.PetalLengthCm[data.Labels == 0],data.PetalWidthCm[data.Labels == 0])
plt.scatter(data.PetalLengthCm[data.Labels == 1],data.PetalWidthCm[data.Labels == 1])
plt.scatter(data.PetalLengthCm[data.Labels == 2],data.PetalWidthCm[data.Labels == 2])
plt.scatter(data.PetalLengthCm[data.Labels == 3],data.PetalWidthCm[data.Labels == 3])


data.drop(["Labels"],axis=1,inplace=True)



plt.subplot(1,5,5)
plt.title('original labels',fontsize=15)
plt.xlabel('PetalLengthCm')
plt.scatter(df.PetalLengthCm[df.Species == 'Iris-setosa'],df.PetalWidthCm[df.Species == 'Iris-setosa'])
plt.scatter(df.PetalLengthCm[df.Species == 'Iris-virginica'],df.PetalWidthCm[df.Species == 'Iris-virginica'])
plt.scatter(df.PetalLengthCm[df.Species == 'Iris-versicolor'],df.PetalWidthCm[df.Species == 'Iris-versicolor'])


plt.subplots_adjust(top=.8)
plt.show()
#data.drop(["Labels"],axis=1,inplace=True)

# kmeans
kmeans = KMeans(n_clusters=3)
kmeans_predict = kmeans.fit_predict(data)

# cross tabulation table for kmeans
df1 = pd.DataFrame({'Labels':kmeans_predict,"Species":df['Species']})
ct1 = pd.crosstab(df1['Labels'],df1['Species'])


# hierarchy
hc_cluster = AgglomerativeClustering(n_clusters=3)
hc_predict = hc_cluster.fit_predict(data)

# cross tabulation table for Hierarchy
df2 = pd.DataFrame({'Labels':hc_predict,"Species":df['Species']})
ct2 = pd.crosstab(df2['Labels'],df2['Species'])


plt.figure(figsize=(24,8))
plt.suptitle("CROSS TABULATIONS",fontsize=18)
plt.subplot(1,2,1)
plt.title("KMeans")
sns.heatmap(ct1,annot=True,cbar=False,cmap="vlag")

plt.subplot(1,2,2)
plt.title("Hierarchy")
sns.heatmap(ct2,annot=True,cbar=False,cmap="pink")

plt.show()
