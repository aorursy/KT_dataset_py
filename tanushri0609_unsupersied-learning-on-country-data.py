#Import the libraries
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb
from sklearn.cluster import KMeans
import numpy as np

#Read the CSV File
data_frame=pd.read_csv('../input/unsupervised-learning-on-country-data/Country-data.csv', sep=',', header=0)
data_frame.head()
# Look for null data
data_frame.info()
# Look at the description of the data
data_frame.describe()
#Visualize the data
sb.pairplot(data_frame[1:], palette='husl')
plt.show()
# find the correlation
sb.heatmap(data_frame[1:].corr(),annot=True)
plt.show()
frame=data_frame.drop('country',axis=1)
frame.head()
# Find number of cluster
SSE = []
for cluster in range(1,20):
    kmeans = KMeans(n_jobs = -1, n_clusters = cluster, init='k-means++')
    kmeans.fit(frame)
    SSE.append(kmeans.inertia_)

# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(1,20), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()
# 3 K-means cluster are making and give label to each cluster
kmeans=KMeans(n_clusters=3)
kmeans.fit(data_frame[['income','gdpp']])
y_means=kmeans.predict(data_frame[['income','gdpp']])
cluster=data_frame
cluster['Labels']=kmeans.labels_
#Divide the cluster into 3 different datafrme
cluster_label0=cluster[cluster['Labels']==0]
cluster_label1=cluster[cluster['Labels']==1]
cluster_label2=cluster[cluster['Labels']==2]
#Plot the cluster
fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(cluster_label0['income'],cluster_label0['gdpp'],color='r',label='cluster 0')
ax.scatter(cluster_label1['income'],cluster_label1['gdpp'],color='b',label='cluster 1')
ax.scatter(cluster_label2['income'],cluster_label2['gdpp'],color='g',label='cluster 2')
ax.legend(labels=['cluster 0','cluster 1','cluster2',])
ax.set_ylabel('gdpp')
ax.set_xlabel('income')
plt.show()
# Select the lowest cluster
df=cluster['Labels'].value_counts()
if(df[0]==128):
    select_frame=cluster_label0
elif(df[1]==128):
    select_frame=cluster_label1
else:
    select_frame=cluster_label2
# Perform K-mean cluster algorithm on the lowest cluster
kmeans=KMeans(n_clusters=3)
kmeans.fit(select_frame[['child_mort','total_fer']])
y_means=kmeans.predict(select_frame[['child_mort','total_fer']])
cluster_1=select_frame
cluster_1['Labels']=kmeans.labels_
cluster_1.describe()
#divide the cluster into 3 different data frame
cluster_label0=cluster_1[cluster_1['Labels']==0]
cluster_label1=cluster_1[cluster_1['Labels']==1]
cluster_label2=cluster_1[cluster_1['Labels']==2]
# Plot the clusters
fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(cluster_label0['child_mort'],cluster_label0['total_fer'],color='r',label='cluster 0')
ax.scatter(cluster_label1['child_mort'],cluster_label1['total_fer'],color='b',label='cluster 1')
ax.scatter(cluster_label2['child_mort'],cluster_label2['total_fer'],color='y',label='cluster 2')
ax.legend(labels=['cluster 0','cluster 1','cluster2'])
ax.set_ylabel('child_mort')
ax.set_xlabel('total_fer')
plt.show()
df=cluster_1['Labels'].value_counts()
if(df[0]==18):
    result=cluster_label0
elif(df[1]==18):
    result=cluster_label1
else:
    result=cluster_label2   
result.sort_values(by=['life_expec'])