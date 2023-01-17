!pip install minisom
!pip install tslearn
# Native libraries
import os
import math
# Essential Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Preprocessing
from sklearn.preprocessing import MinMaxScaler
# Algorithms
from minisom import MiniSom
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
directory = '/kaggle/input/retail-and-retailers-sales-time-series-collection/'

mySeries = []
namesofMySeries = []
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        df = pd.read_csv(directory+filename)
        df = df.loc[:,["date","value"]]
        # While we are at it I just filtered the columns that we will be working on
        df.set_index("date",inplace=True)
        # ,set the date columns as index
        df.sort_index(inplace=True)
        # and lastly, ordered the data according to our date index
        mySeries.append(df)
        namesofMySeries.append(filename[:-4])
print(len(mySeries))
fig, axs = plt.subplots(6,4,figsize=(25,25))
fig.suptitle('Series')
for i in range(6):
    for j in range(4):
        if i*4+j+1>len(mySeries): # pass the others that we can't fill
            continue
        axs[i, j].plot(mySeries[i*4+j].values)
        axs[i, j].set_title(namesofMySeries[i*4+j])
plt.show()
fig, axs = plt.subplots(6,4,figsize=(25,25))
fig.suptitle('Series')
for i in range(6):
    for j in range(4):
        if i*4+j+1>len(mySeries): # pass the others that we can't fill
            continue
        axs[i, j].plot(mySeries[i*4+j].values)
        axs[i, j].set_title(namesofMySeries[i*4+j])
plt.show()
series_lengths = {len(series) for series in mySeries}
print(series_lengths)
ind = 0
for series in mySeries:
    print("["+str(ind)+"] "+series.index[0]+" "+series.index[len(series)-1])
    ind+=1
max_len = max(series_lengths)
longest_series = None
for series in mySeries:
    if len(series) == max_len:
        longest_series = series
problems_index = []

for i in range(len(mySeries)):
    if len(mySeries[i])!= max_len:
        problems_index.append(i)
        mySeries[i] = mySeries[i].reindex(longest_series.index)
def nan_counter(list_of_series):
    nan_polluted_series_counter = 0
    for series in list_of_series:
        if series.isnull().sum().sum() > 0:
            nan_polluted_series_counter+=1
    print(nan_polluted_series_counter)
nan_counter(mySeries)
for i in problems_index:
    mySeries[i].interpolate(limit_direction="both",inplace=True)
nan_counter(mySeries)
a = [[2],[7],[11],[14],[19],[23],[26]]
b = [[20000000],[40000000],[60000000],[80000000],[100000000],[120000000],[140000000]]
fig, axs = plt.subplots(1,3,figsize=(25,5))
axs[0].plot(a)
axs[0].set_title("Series 1")
axs[1].plot(b)
axs[1].set_title("Series 2")
axs[2].plot(a)
axs[2].plot(b)
axs[2].set_title("Series 1 & 2")
plt.figure(figsize=(25,5))
plt.plot(MinMaxScaler().fit_transform(a))
plt.plot(MinMaxScaler().fit_transform(b))
plt.title("Normalized Series 1 & Series 2")
plt.show()
for i in range(len(mySeries)):
    scaler = MinMaxScaler()
    mySeries[i] = MinMaxScaler().fit_transform(mySeries[i])
    mySeries[i]= mySeries[i].reshape(len(mySeries[i]))
print("max: "+str(max(mySeries[0]))+"\tmin: "+str(min(mySeries[0])))
print(mySeries[0][:5])
a = [1,2]
b = [3,7]
c = [1,3]
d = [3,8]
img = plt.imread("/kaggle/input/notebook-material/arrow.png")
fig, axs = plt.subplots(1,3,figsize=(25,5))
axs[0].plot(a)
axs[0].plot(b)
axs[0].plot(c)
axs[0].plot(d)
axs[0].set_title("Time Series")
axs[1].imshow(img)
axs[1].axis("off")
axs[2].set_title("Data Points")
axs[2].scatter(a[0],a[1], s=300)
axs[2].scatter(b[0],b[1], s=300)
axs[2].scatter(c[0],c[1], s=300)
axs[2].scatter(d[0],d[1], s=300)
plt.show()
som_x = som_y = math.ceil(math.sqrt(math.sqrt(len(mySeries))))
# I didn't see its significance but to make the map square,
# I calculated square root of map size which is 
# the square root of the number of series
# for the row and column counts of som

som = MiniSom(som_x, som_y,len(mySeries[0]), sigma=0.3, learning_rate = 0.1)

som.random_weights_init(mySeries)
som.train(mySeries, 50000)

win_map = som.win_map(mySeries)
# Returns the mapping of the winner nodes and inputs

fig, axs = plt.subplots(som_x,som_y,figsize=(25,25))
fig.suptitle('Clusters')
row_i=0
column_j=0
for cluster in win_map.keys():
        for series in win_map[cluster]:
            axs[row_i, column_j].plot(series,c="gray",alpha=0.5)
        axs[row_i, column_j].plot(np.average(np.vstack(win_map[cluster]),axis=0),c="red")
        axs[row_i, column_j].set_title("Cluster "+str(row_i*som_y+column_j))
        column_j+=1
        if column_j%som_y == 0:
            row_i+=1
            column_j=0
        
plt.show()
win_map = som.win_map(mySeries)
# Returns the mapping of the winner nodes and inputs

fig, axs = plt.subplots(som_x,som_y,figsize=(25,25))
fig.suptitle('Clusters')
row_i=0
column_j=0
for cluster in win_map.keys():
        for series in win_map[cluster]:
            axs[row_i, column_j].plot(series,c="gray",alpha=0.5)
        axs[row_i, column_j].plot(np.average(np.vstack(win_map[cluster]),axis=0),c="red")
        axs[row_i, column_j].set_title("Cluster "+str(row_i*som_y+column_j))
        column_j+=1
        if column_j%som_y == 0:
            row_i+=1
            column_j=0
        
plt.show()
win_map = som.win_map(mySeries)

fig, axs = plt.subplots(som_x,som_y,figsize=(25,25))
fig.suptitle('Clusters')
row_i=0
column_j=0
for cluster in win_map.keys():
        for series in win_map[cluster]:
            axs[row_i, column_j].plot(series,c="gray",alpha=0.4)
        axs[row_i, column_j].plot(dtw_barycenter_averaging(np.vstack(win_map[cluster])),c="red")
        axs[row_i, column_j].set_title("Cluster "+str(row_i*som_y+column_j))
        column_j+=1
        if column_j%som_y == 0:
            row_i+=1
            column_j=0
        
plt.show()
win_map = som.win_map(mySeries)

fig, axs = plt.subplots(som_x,som_y,figsize=(25,25))
fig.suptitle('Clusters')
row_i=0
column_j=0
for cluster in win_map.keys():
        for series in win_map[cluster]:
            axs[row_i, column_j].plot(series,c="gray",alpha=0.4)
        axs[row_i, column_j].plot(dtw_barycenter_averaging(np.vstack(win_map[cluster])),c="red")
        axs[row_i, column_j].set_title("Cluster "+str(row_i*som_y+column_j))
        column_j+=1
        if column_j%som_y == 0:
            row_i+=1
            column_j=0
        
plt.show()
cluster_c = [len(win_map[cluster]) for cluster in win_map.keys()]
cluster_n = ["cluster_"+str(i) for i in range(len(win_map.keys()))]
plt.figure(figsize=(25,5))
plt.title("Cluster Distribution for SOM")
plt.bar(cluster_n,cluster_c)
plt.show()
cluster_count = math.ceil(math.sqrt(len(mySeries))) 
# A good rule of thumb is choosing k as the square root of the number of points in the training data set in kNN

km = TimeSeriesKMeans(n_clusters=cluster_count, metric="dtw")

labels = km.fit_predict(mySeries)
plot_count = math.ceil(math.sqrt(cluster_count))

fig, axs = plt.subplots(plot_count,plot_count,figsize=(25,25))
fig.suptitle('Clusters')
row_i=0
column_j=0
# For each label there is,
# plots every series with that label
for label in set(labels):
    cluster = []
    for i in range(len(labels)):
            if(labels[i]==label):
                axs[row_i, column_j].plot(mySeries[i],c="gray",alpha=0.4)
                cluster.append(mySeries[i])
    if len(cluster) > 0:
        axs[row_i, column_j].plot(np.average(np.vstack(cluster),axis=0),c="red")
    axs[row_i, column_j].set_title("Cluster "+str(row_i*som_y+column_j))
    column_j+=1
    if column_j%plot_count == 0:
        row_i+=1
        column_j=0
        
plt.show()
plot_count = math.ceil(math.sqrt(cluster_count))

fig, axs = plt.subplots(plot_count,plot_count,figsize=(25,25))
fig.suptitle('Clusters')
row_i=0
column_j=0
# For each label there is,
# plots every series with that label
for label in set(labels):
    cluster = []
    for i in range(len(labels)):
            if(labels[i]==label):
                axs[row_i, column_j].plot(mySeries[i],c="gray",alpha=0.4)
                cluster.append(mySeries[i])
    if len(cluster) > 0:
        axs[row_i, column_j].plot(np.average(np.vstack(cluster),axis=0),c="red")
    axs[row_i, column_j].set_title("Cluster "+str(row_i*som_y+column_j))
    column_j+=1
    if column_j%plot_count == 0:
        row_i+=1
        column_j=0
        
plt.show()
plot_count = math.ceil(math.sqrt(cluster_count))

fig, axs = plt.subplots(plot_count,plot_count,figsize=(25,25))
fig.suptitle('Clusters')
row_i=0
column_j=0
for label in set(labels):
    cluster = []
    for i in range(len(labels)):
            if(labels[i]==label):
                axs[row_i, column_j].plot(mySeries[i],c="gray",alpha=0.4)
                cluster.append(mySeries[i])
    if len(cluster) > 0:
        axs[row_i, column_j].plot(dtw_barycenter_averaging(np.vstack(cluster)),c="red")
    axs[row_i, column_j].set_title("Cluster "+str(row_i*som_y+column_j))
    column_j+=1
    if column_j%plot_count == 0:
        row_i+=1
        column_j=0
        
plt.show()
cluster_c = [len(labels[labels==i]) for i in range(cluster_count)]
cluster_n = ["cluster_"+str(i) for i in range(cluster_count)]
plt.figure(figsize=(15,5))
plt.title("Cluster Distribution for KMeans")
plt.bar(cluster_n,cluster_c)
plt.show()
pca = PCA(n_components=2)

mySeries_transformed = pca.fit_transform(mySeries)
plt.figure(figsize=(25,10))
plt.scatter(mySeries_transformed[:,0],mySeries_transformed[:,1], s=300)
plt.show()
print(mySeries_transformed[0:5])
kmeans = KMeans(n_clusters=cluster_count,max_iter=5000)

labels = kmeans.fit_predict(mySeries_transformed)
plt.figure(figsize=(25,10))
plt.scatter(mySeries_transformed[:, 0], mySeries_transformed[:, 1], c=labels, s=300)
plt.show()
plot_count = math.ceil(math.sqrt(cluster_count))

fig, axs = plt.subplots(plot_count,plot_count,figsize=(25,25))
fig.suptitle('Clusters')
row_i=0
column_j=0
for label in set(labels):
    cluster = []
    for i in range(len(labels)):
            if(labels[i]==label):
                axs[row_i, column_j].plot(mySeries[i],c="gray",alpha=0.4)
                cluster.append(mySeries[i])
    if len(cluster) > 0:
        axs[row_i, column_j].plot(np.average(np.vstack(cluster),axis=0),c="red")
    axs[row_i, column_j].set_title("Cluster "+str(row_i*som_y+column_j))
    column_j+=1
    if column_j%plot_count == 0:
        row_i+=1
        column_j=0
        
plt.show()
cluster_c = [len(labels[labels==i]) for i in range(cluster_count)]
cluster_n = ["cluster_"+str(i) for i in range(cluster_count)]
plt.figure(figsize=(15,5))
plt.title("Cluster Distribution for KMeans")
plt.bar(cluster_n,cluster_c)
plt.show()