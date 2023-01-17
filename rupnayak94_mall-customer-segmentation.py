import os

import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt

from matplotlib import style

from matplotlib.ticker import MaxNLocator

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score, silhouette_samples

from mpl_toolkits.mplot3d import Axes3D
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        path=os.path.join(dirname, filename)

        print(path)
raw_data=pd.read_csv(path)

raw_data
raw_data.drop(["CustomerID"], axis=1, inplace=True)
raw_data.columns
raw_data.info()
raw_data.describe()
raw_data.isnull().sum()
gender_group_mean=raw_data.groupby(["Gender"]).mean().reset_index()

gender_group_mean
plt.figure(figsize=(6,6))

plt.bar(gender_group_mean["Gender"], gender_group_mean["Annual Income (k$)"], color=["orange","indigo"])
plt.figure(figsize=(8,6))

plt.bar(gender_group_mean["Gender"], gender_group_mean["Spending Score (1-100)"], color=["b","g"])
plt.figure(figsize=(6,6))

sns.boxenplot(raw_data["Gender"], raw_data["Annual Income (k$)"], palette ="rainbow")

plt.grid(color="silver")
plt.figure(figsize=(14,6))



plt.subplot(1,2,1) #subplot rows, columns and Index

sns.boxenplot(raw_data["Gender"], raw_data["Spending Score (1-100)"], palette ="rocket")

plt.grid(color="silver")



plt.subplot(1,2,2)

sns.boxplot(raw_data["Gender"], raw_data["Spending Score (1-100)"], palette ="seismic")

plt.grid(color="silver")
plt.figure(figsize=(20,8))



plt.subplot(1,2,1)

ax=sns.distplot(raw_data["Spending Score (1-100)"], color="mediumspringgreen")



plt.subplot(1,2,2)

ax=sns.distplot(raw_data["Annual Income (k$)"], color="blue")

raw_data.groupby(["Gender"]).count().reset_index()
raw_data["Gender"]=raw_data["Gender"].replace(["Male", "Female"], [0,1])

raw_data
std_slr=StandardScaler()

raw_data_std=std_slr.fit_transform(raw_data)

raw_data_std[0:5]
raw_data_std=pd.DataFrame(raw_data_std, columns=raw_data.columns)

raw_data_std
data=raw_data_std.iloc[:,2:].values
wss=[]

sil=[]

for k in range(2,20):

    kmeans=KMeans(n_clusters=k, random_state=1).fit(data)

    wss.append(kmeans.inertia_)

    labels=kmeans.labels_

    sil.append(silhouette_score(raw_data_std, labels, metric = 'euclidean'))

print(wss)

print(sil)
k=range(2,20)

fig,ax=plt.subplots(figsize=(20,8))

ax.set_facecolor('white')

ax.plot(k,wss, color="green")

ax.xaxis.set_major_locator(MaxNLocator(nbins=20,integer=True)) #forces the scales to be in integer, nbins as scales size.

ax.set_xlabel("No of clusters",color="black", fontsize=20)

ax.set_ylabel("WSS (With in sum of squares)", color="green", fontsize=20)

ax2=ax.twinx() #creates the second axis on the first plot (ax)

ax2.plot(k,sil, color="blue")

ax2.set_ylabel("Silhouette scores", color="Blue", fontsize=20)

ax.grid(True, color="silver")

plt.show()
n=5

kmeans=KMeans(n_clusters=n, random_state=1).fit(data)

clusters=kmeans.labels_

centroids=kmeans.cluster_centers_

clusters
raw_data_std2=pd.DataFrame(pd.concat([raw_data_std, pd.Series(kmeans.labels_)], axis=1).rename(columns={0:"Clusters"})).copy()

raw_data2=pd.DataFrame(pd.concat([raw_data, pd.Series(kmeans.labels_)], axis=1).rename(columns={0:"Clusters"})).copy()

raw_data2
raw_data2_copy=raw_data2.copy()

raw_data2_copy.sort_values(["Clusters"], inplace=True)

raw_data_std2.sort_values(["Clusters"], inplace=True)

for i in range(0,n+1):

    raw_data2_copy["Clusters"]=raw_data2_copy["Clusters"].replace(i, chr(i+65))

    raw_data_std2["Clusters"]=raw_data_std2["Clusters"].replace(i, chr(i+65))

raw_data2_copy["Clusters"].unique()
raw_data2_copy
fig=plt.figure(figsize=(14,8))

ax=Axes3D(fig) #used to create 3D plots, part of matplotlib

x=np.array(raw_data["Age"])

y=np.array(raw_data["Annual Income (k$)"])

z=np.array(raw_data["Spending Score (1-100)"])

centroids=np.array(centroids)

ax.scatter(x, y,z, c=y)

plt.title('SPENDING SCORE VS ANNUAL INCOME VS AGE')

ax.set_xlabel('Age')

ax.set_ylabel('ANNUAL Income')

ax.set_zlabel('Spending Score')
x=raw_data.iloc[:,2:].values

y_means=kmeans.fit_predict(data)

print(y_means)

print("X shape:",x.shape)
plt.figure(figsize=(14,8))



plt.scatter(x[y_means==0,0], x[y_means==0,1], color="cyan", label="Normal") #pos argument 0 for X and 1 for Y

plt.scatter(x[y_means==1,0], x[y_means==1,1], color="indigo", label="High Spenders") #y_means==1 for cluster center 1

plt.scatter(x[y_means==2,0], x[y_means==2,1], color="red", label="Value buyers")

plt.scatter(x[y_means==3,0], x[y_means==3,1], color="green", label="Savers")

plt.scatter(x[y_means==4,0], x[y_means==4,1], color="blue", label="Impulse buyers")



plt.xlabel("Annual Income (Thousand $)")

plt.ylabel("Spending Score (1-100)")

plt.legend(loc="right")

plt.grid(True, color="silver")

plt.title("Customers Demographics")

plt.show()
plt.figure(figsize=(14,8))

sns.scatterplot("Annual Income (k$)", "Spending Score (1-100)", hue="Clusters", data=raw_data_std2)

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:, 1], s = 100,c="Blue", label = 'centeroid')

plt.legend()

plt.grid()
sample_silhouette_values=silhouette_samples(raw_data_std, clusters)

raw_data_std2["silhouette_values"]=sample_silhouette_values
raw_data_std2
raw_data_std2.groupby(["Clusters"])["silhouette_values"].mean()
silhouette_score(raw_data_std, clusters) #all data points