%matplotlib inline

from __future__ import print_function



try:

    xrange

except NameError:

    xrange = range

import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, MiniBatchKMeans

from sklearn.metrics import silhouette_samples, silhouette_score

from sklearn.preprocessing import StandardScaler, MaxAbsScaler
def generate_2dim_normal(mean,variance,covariance,sample_size):

    cov = [[variance,covariance],[covariance,variance]]

    return np.random.multivariate_normal(mean,cov,sample_size)

cluster1 = generate_2dim_normal(mean = [0,8],variance=1,covariance=0,sample_size=500)

cluster2 = generate_2dim_normal(mean = [-1,0],variance=1,covariance=0,sample_size=500)

cluster3 = generate_2dim_normal(mean = [10,10],variance=1,covariance=0,sample_size=300)

cluster4 = generate_2dim_normal(mean = [5,5.5],variance=0.8,covariance=-0.1,sample_size=200)

data = np.vstack((cluster1,cluster2,cluster3,cluster4))
fig = plt.figure()

ax = fig.add_subplot(1,1,1)

ax.scatter(data[:,0],data[:,1])

ax.set_title("scatter plot")

ax.set_xlabel("X1")

ax.set_ylabel("X2")
km = KMeans(n_clusters=4,init="k-means++",n_init=10,max_iter=300)

##Init is a sheet argument,max_iter is the max times to update the sheet

km.fit(data)

cluster_labels = km.predict(data)
fig = plt.figure()

ax = fig.add_subplot(1,1,1)

colorlist =["tomato","antiquewhite","blueviolet","cornflowerblue","darkgreen","seashell","skyblue","mediumseagreen"]



#The unique number of cluster_ids is 「0,1,2,3」

cluster_ids = list(set(cluster_labels))



for k in range(len(cluster_ids)):

    cluster_id = cluster_ids[k]

    label_ = "cluster = %d" % cluster_id

    data_by_cluster = data[cluster_labels == cluster_id]

    ax.scatter(data_by_cluster[:,0],data_by_cluster[:,1],c=colorlist[k],label=label_)



ax.set_title("Clustering")

ax.set_xlabel("X1")

ax.set_ylabel("Y1")

ax.legend(loc="lower right")
#elbow method

max_cluster = 10

clusters = range(1,max_cluster)

intra_sum_of_square_list = []

for k in clusters:

    km = KMeans(n_clusters=k,init="k-means++",n_init=10,max_iter=300)

    km.fit(data)

    intra_sum_of_square_list.append(km.inertia_)

#You can get wcss value. through [inertia_]
fig = plt.figure()

ax = fig.add_subplot(1,1,1)

ax.set_title("Elbow Method")

ax.set_xlabel("Number of Cluster")

ax.set_ylabel("Intra sum of distances(WCSS)")

plt.plot(clusters,intra_sum_of_square_list)
#Silhouette blot (silhouette score - 1 to 1 points, close to 1, close to the center of the cluster, 

#conversely close to the center of the next cluster)

n_clusters=4

km = KMeans(n_clusters=4,init="k-means++",n_init=10,max_iter=300)

km.fit(data)

cluster_labels = km.predict(data)



#Calculate the average of silhouette scores

silhouette_avg = silhouette_score(data,cluster_labels)



#Calculate the silhouette score for each data

each_silhouette_score = silhouette_samples(data,cluster_labels,metric="euclidean")
#Visualization

fig =plt.figure()

ax = fig.add_subplot(1,1,1)

y_lower =10

for i in range(n_clusters):

    ith_cluster_silhouette_values = each_silhouette_score[cluster_labels == i]

    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]

    y_upper = y_lower + size_cluster_i

    

    color = colorlist[i]

    ax.fill_betweenx(np.arange(y_lower,y_upper),0,ith_cluster_silhouette_values,facecolor=color,edgecolor=color,alpha=0.3)

    

    #label the silhouse plots with their cluster numbers at the middle

    ax.text(-0.05,y_lower + 0.5 * size_cluster_i,str(i))

    

    #compute the new y_lower for next plot

    y_lower = y_upper +10 

    

ax.set_title("Silhuoette plot")

ax.set_xlabel("silhouette score")

ax.set_ylabel("Cluster label")

    

#the vertical line for average silhouette score of all the values

ax.axvline(x=silhouette_avg,color="red",linestyle="--")

    

ax.set_yticks([])

ax.set_xticks([-0.2,0,0.2,0.4,0.6,0.8,1])

    
df = pd.read_csv("../input/Wholesale_customers_data.csv")

                 
df.head()
df.describe()
#due to the category[channel,region]has the category values,not the continuity values,so, we should devide them first

cols =["Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen"]

dataframe = df[cols]

dataframe.head()

#standardScaler

scaler = MaxAbsScaler()

#min=0,MAX=1



#scaler = StandardScaler()

#mean=0,arufa=1



dataset = scaler.fit_transform(dataframe)
#Elbow method

max_cluster = 10

clusters = range(1,max_cluster)

intra_sum_of_square_list = []

for k in clusters:

    km = KMeans(n_clusters=k,init="k-means++",n_init=10,max_iter=300)

    km.fit(dataset)

    intra_sum_of_square_list.append(km.inertia_)

fig = plt.figure()

ax = fig.add_subplot(1,1,1)

ax.set_title("Elbow Method")

ax.set_xlabel("Number of Cluster")

ax.set_ylabel("Intra sum of distances(WCSS)")

plt.plot(clusters,intra_sum_of_square_list)

    
#Silhouette Plot

n_clusters=6

km = KMeans(n_clusters=6,init="k-means++",n_init=10,max_iter=300)

km.fit(dataset)

cluster_labels = km.predict(dataset)



#Calculate the average of silhouette scores

silhouette_avg = silhouette_score(dataset,cluster_labels)



#Calculate the silhouette score for each data

each_silhouette_score = silhouette_samples(dataset,cluster_labels,metric="euclidean")

fig =plt.figure()

ax = fig.add_subplot(1,1,1)

y_lower =10

for i in range(n_clusters):

    ith_cluster_silhouette_values = each_silhouette_score[cluster_labels == i]

    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]

    y_upper = y_lower + size_cluster_i

    

    color = colorlist[i]

    ax.fill_betweenx(np.arange(y_lower,y_upper),0,ith_cluster_silhouette_values,facecolor=color,edgecolor=color,alpha=0.3)

    

    #label the silhouse plots with their cluster numbers at the middle

    ax.text(-0.05,y_lower + 0.5 * size_cluster_i,str(i))

    

    #compute the new y_lower for next plot

    y_lower = y_upper +10 

    

ax.set_title("Silhuoette plot")

ax.set_xlabel("Silhouette score")

ax.set_ylabel("Cluster label")

    

#the vertical line for average silhouette score of all the values

ax.axvline(x=silhouette_avg,color="red",linestyle="--")

    

ax.set_yticks([])

ax.set_xticks([-0.2,0,0.2,0.4,0.6,0.8,1])
#we can see there are a lot of points at the cluster of number 0.

#but it is hard to say the clustering has been devided perfectly.

#Let us see the average of  each cluster at last.
km_centers =pd.DataFrame(km.cluster_centers_,columns=cols)

km_centers.plot.bar(ylim=[0,2],fontsize=10)

km_centers