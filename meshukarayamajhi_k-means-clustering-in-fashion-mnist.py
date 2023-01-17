import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cv2

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df_test = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')
df_train = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')

df_train_ = df_train.drop(['label'],axis=1).to_numpy()
#df_train.head()
print(df_train_.shape) 

#converting grayscale images to binary
bw_df_train = np.zeros((60000,784))
bw_df_train = np.array(bw_df_train)
for index, value in enumerate(df_train_):
    value = np.array(value, dtype='float32') #Threshold method only works on 8-bit integer/32 bit floating point arrays
    ret,bw = cv2.threshold(value,127,255,cv2.THRESH_BINARY) 
    bw= bw.reshape(784)
    bw_df_train[index]=bw
#print(bw.shape)
print(type(df_train_))
print(type(bw_df_train))
print(bw_df_train.shape)
print(type(df_train))
#print(bw_df_train[1])
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler();
#fitting training set; .fit is used to fit/train data frequently and quickly
#seperating label from rest of the features
bw_df_train_nl = df_train.drop(['label'],axis=1)
bw_df_test_nl = df_test.drop(['label'],axis=1)
scaler.fit(bw_df_train_nl)
#transform or map in both test and train set
df_train_nl_no_pca = scaler.transform(bw_df_train_nl)
df_test_nl_no_pca = scaler.transform(bw_df_test_nl)
#speeds up ml algorithm by selecting principal components
#importing and applying PCA (principal component algorithm)
from sklearn.decomposition import PCA
pca= PCA(.98)
#.95 tells the algorithm to choose principal components such that 95% varaince of dataset is retained
bw_df_train_nl = df_train_nl_no_pca
pca.fit(bw_df_train_nl)
print(pca.n_components_)

bw_df_train_nl = pca.transform(bw_df_train_nl)
bw_df_test_nl = pca.transform(bw_df_test_nl)
inverse = pca.inverse_transform(bw_df_train_nl)

print(bw_df_train.shape)
plt.figure(figsize= (10,6))

#displaying original image in first plot of 1*2 grid
plt.subplot(1,2,1)
#reshaping from 1D to 2D array (28 * 28 pixels) which can be plotted 
plt.imshow(bw_df_train[500].reshape(28, 28), cmap= plt.cm.gray)
plt.title('Original Image', fontsize = 18)
#displaying inverse tranformed image 
plt.subplot(1,2,2)
plt.imshow(inverse[500].reshape(28,28),cmap= plt.cm.gray)
plt.title('Inversed Image', fontsize = 18)
#using kmeans clustering with 420 features
import time
from sklearn.cluster import KMeans
start_time = time.time()
#np.random.seed(0)
kmeans = KMeans (init = "k-means++", n_clusters=10, n_init=35)
kmeans.fit(bw_df_train_nl)
print("using kmeans clustering with 420 features took %s seconds" % (time.time() - start_time))
#using kmeans clustering with 784 features
start_time = time.time()
kmeans1 = KMeans (init = "k-means++", n_clusters=10, n_init=35)
kmeans1.fit(df_train_nl_no_pca)
print("using kmeans clustering with 784 features took %s seconds" % (time.time() - start_time))
#attributes of KMeans() with PCA
kmeans_cluster_labels = kmeans.labels_
print(kmeans_cluster_labels)
print(len(kmeans_cluster_labels))
print("The labels of the clusters (with pca) are " + str(np.unique(kmeans_cluster_labels)))
#attributes of KMeans() without PCA
kmeans1_cluster_labels = kmeans1.labels_
print(kmeans1_cluster_labels)
print(len(kmeans1_cluster_labels))
print("The labels of the clusters (without pca) are " + str(np.unique(kmeans1_cluster_labels)))
#defining actual number of labels of train dataset
label_column = df_train['label']
print(label_column.shape)
labels = len(np.unique(label_column))
print(labels)
label_names = {0:'T-shirt/top', 1:'Trouser',2: 'Pullover',3: 'Dress',4: 'Coat',5:
               'Sandal',6: 'Shirt', 7:'Sneaker',8:  'Bag',9: 'Ankle boot'} #acc to dataset
#print(label_names.values())
num_cluster_labels = len(np.unique(kmeans_cluster_labels))
#assigning indexes into respective cluster members
cluster_indexes = [[] for i in range(labels)]
#print(cluster_indexes)
for i,label in enumerate(kmeans_cluster_labels):
    for n in range(num_cluster_labels):
        if label == n:
            cluster_indexes[n].append(i)
        else:
            continue
#number of datapoints in individual clusters
print('With PCA')
for i in range(num_cluster_labels):
    print('No. of items in Cluster ' + str(i) + ': ' + str(len(cluster_indexes[i])))
num_cluster_labels_no_pca = len(np.unique(kmeans1_cluster_labels))
#assigning indexes into respective cluster members
cluster_indexes_no_pca = [[] for i in range(labels)]
#print(cluster_indexes)
for i,label in enumerate(kmeans1_cluster_labels):
    for n in range(num_cluster_labels_no_pca):
        if label == n:
            cluster_indexes_no_pca[n].append(i)
        else:
            continue
#number of datapoints in individual clusters
print('Without PCA')
for i in range(num_cluster_labels_no_pca):
    print('No. of items in Cluster ' + str(i) + ': ' + str(len(cluster_indexes_no_pca[i])))
import plotly.graph_objs as go
#import plotly.express as px
from plotly.offline import iplot
#plt.rcParams["figure.figsize"] = (20,25)
trace =[[] for i in range(0,10)]
colors = ['red','green' ,'blue','purple','magenta','yellow','cyan','maroon','teal','black']
#data_info = 'sandal'
for i in range(0,10):
    my_members = (cluster_indexes[i])
    num_cluster = i
    trace[i] = go.Scatter3d(
        x=bw_df_train_nl[my_members,0],
        y=bw_df_train_nl[my_members,1],
        z=bw_df_train_nl[my_members,2],
        mode='markers',
        marker = dict(size = 2,color = colors[i]),
        name ='Cluster'+str(i),
        hoverinfo = 'text',
        text = 'Cluster:' + str(num_cluster) # + ' Item: ' + data_info 
    )
    layout = go.Layout(title = '3D Scatter plot with 420 Principal Components')
fig = go.Figure(data=[trace[0],trace[1],trace[2],trace[3],trace[4],trace[5],trace[6],trace[7],trace[8],trace[9]],layout=layout)
iplot(fig)
#3D plot with two principal components
#two principal components from 784 features
final_arr = scaler.fit_transform(df_train_)
pca1= PCA(n_components=3)
two_pc_arr = pca1.fit_transform(final_arr)
trace =[[] for i in range(0,10)]
colors = ['red','green' ,'blue','purple','magenta','yellow','cyan','maroon','teal','black']
#data_info = 'sandal'
for i in range(0,10):
    my_members = (cluster_indexes[i])
    num_cluster = i
    trace[i] = go.Scatter3d(
        x=two_pc_arr[my_members,0],
        y=two_pc_arr[my_members,1],
        z=two_pc_arr[my_members,2],
        mode='markers',
        marker = dict(size = 2,color = colors[i]),
        name ='Cluster'+str(i),
        hoverinfo = 'text',
        text = 'Cluster:' + str(num_cluster) # + ' Item: ' + data_info 
    )
    layout = go.Layout(title = '3D Scatter plot with 2 Principal Components')
fig = go.Figure(data=[trace[0],trace[1],trace[2],trace[3],trace[4],trace[5],trace[6],trace[7],trace[8],trace[9]],layout=layout)
iplot(fig)
#import plotly.graph_objs as go
#import plotly.express as px
#from plotly.offline import iplot
#plt.rcParams["figure.figsize"] = (20,25)
trace =[[] for i in range(0,10)]
colors = ['red','green' ,'blue','purple','magenta','yellow','cyan','maroon','teal','black']
#data_info = 'sandal'
for i in range(0,10):
    my_members = (cluster_indexes_no_pca[i])
    num_cluster = i
    trace[i] = go.Scatter3d(
        x=df_train_nl_no_pca[my_members,110],
        y=df_train_nl_no_pca[my_members,160],
        z=df_train_nl_no_pca[my_members,200],
        mode='markers',
        marker = dict(size = 2,color = colors[i]),
        name ='Cluster'+str(i),
        hoverinfo = 'text',
        text = 'Cluster:' + str(num_cluster) # + ' Item: ' + data_info 
    )
    layout = go.Layout(title = '3D Scatter plot without PCA')
fig = go.Figure(data=[trace[0],trace[1],trace[2],trace[3],trace[4],trace[5],trace[6],trace[7],trace[8],trace[9]],layout=layout)
iplot(fig)
#assigning actual label values(from train dataset) along side the indices of items of each cluster
Y_label_values = [ [] for i in range(labels)]
for i in range(labels):
    Y_label_values[i]= label_column[cluster_indexes[i]] 
print(Y_label_values[0])
#print(type(Y_label_values))
for index, label in enumerate(Y_label_values):
    print('Index='+ str(index))
    print('Label=' + str(label))
#assigning actual label values(from train dataset) along side the indices of items of each cluster
Y_label_values_no_pca = [ [] for i in range(labels)]
for i in range(labels):
    Y_label_values_no_pca[i]= label_column[cluster_indexes_no_pca[i]] 
print(Y_label_values_no_pca[0])
#print(type(Y_label_values))
for index, label in enumerate(Y_label_values_no_pca):
    print('Index='+ str(index))
    print('Label=' + str(label))
#number of dataitems of a certain category in each cluster
label_count = [[] for i in range(labels)]
for index, label in enumerate(Y_label_values):
    unique,count = np.unique(label, return_counts = True) #returns the individual label values and counts as lists
    label_count[index]= dict(zip(unique,count)) #converts two lists into a dictionary
print(label_count[0])   
print(label_count[1])   
#number of dataitems of a certain category in each cluster
label_count_no_pca = [[] for i in range(labels)]
for index, label in enumerate(Y_label_values_no_pca):
    unique,count = np.unique(label, return_counts = True) #returns the individual label values and counts as lists
    label_count_no_pca[index]= dict(zip(unique,count)) #converts two lists into a dictionary
print(label_count_no_pca[0])   
print(label_count_no_pca[1])   
#Visualization of cluster points using barplots
print('Cluster visualization with PCA')
plt.figure(figsize=(18,20))
for i in range(1,11):
    plt.subplot(5,2,i)
    label_count_=label_count[i-1]
    plt.bar(range(len(label_count_)),list(label_count_.values()),align='center')
    plt.title('Cluster' + str(i-1))
    a=[]
    for key,value in label_count_.items():
        for key1, value1 in label_names.items():
            if key1 == key:
                a.append(value1)
                break
    plt.xticks(range(len(a)),a)       

#Visualization of cluster points using barplots
print('Cluster Visualization without PCA')
plt.figure(figsize=(18,20))
for i in range(1,11):
    plt.subplot(5,2,i)
    label_count_=label_count_no_pca[i-1]
    plt.bar(range(len(label_count_)),list(label_count_.values()),align='center')
    plt.title('Cluster' + str(i-1))
    a=[]
    for key,value in label_count_.items():
        for key1, value1 in label_names.items():
            if key1 == key:
                a.append(value1)
                break
    plt.xticks(range(len(a)),a)       

print("Not much of a drastic difference is seen in variations within clusters with and without PCA.")

print("With PCA, scatter plots are possible.")
print("Execution time of kmeans function on a dataset using pca is faster than the one which uses all the features.")
#visualization for clusters
plt.figure(figsize= (15,20))
cluster = 0
for i in range(1,100):
    plt.subplot(10,10,i)
    plt.imshow(bw_df_train[cluster_indexes[cluster][i+280]].reshape(28,28),cmap= plt.cm.gray)
plt.show()
#visualization for clusters
plt.figure(figsize= (15,20))
cluster = 1
for i in range(1,100):
    plt.subplot(10,10,i)
    plt.imshow(df_train_[cluster_indexes[cluster][i+280]].reshape(28,28),cmap= plt.cm.gray)
plt.show()