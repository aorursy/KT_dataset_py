import random 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets.samples_generator import make_blobs 
%matplotlib inline
np.random.seed(0)
X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.7)

print (X [0:5])
print ('\n')
print (y [0:5])
plt.scatter(X[:, 0], X[:, 1], marker='.')
k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)
k_means.fit(X)
k_means_labels = k_means.labels_
k_means_labels
k_means_cluster_centers = k_means.cluster_centers_
k_means_cluster_centers
# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(6, 4))

# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the
# unique labels.
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# Create a plot
ax = fig.add_subplot(1, 1, 1)

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.
for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):

    # Create a list of all data points, where the data poitns that are 
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means_labels == k)
    
    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]
    
    # Plots the datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    
    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

# Title of the plot
ax.set_title('KMeans')

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())

# Show the plot
plt.show()
# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(6, 4))

# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the
# unique labels.
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# Create a plot
ax = fig.add_subplot(1, 1, 1)

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.
for k, col in zip(range(3), colors):

    # Create a list of all data points, where the data poitns that are 
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means_labels == k)
    
    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]
    
    # Plots the datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    
    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

# Title of the plot
ax.set_title('KMeans')

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())

# Show the plot
plt.show()


url = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/Cust_Segmentation.csv'
import pandas as pd
cust_df = pd.read_csv(url)
cust_df.tail()
cust_df.shape
df = cust_df.drop('Address', axis=1)
df.head()
round (df.describe (),2)
import seaborn as sns
plt.figure (figsize = (15,6))
plt.subplot (2,4,1)
sns.distplot (df.Age, hist = False)

plt.subplot (2,4,2)
sns.distplot (df ['Years Employed'], hist = False)

plt.subplot (2,4,3)
sns.distplot (df ['Income'], hist = False)

plt.subplot (2,4,4)
sns.distplot (df ['Card Debt'], hist = False)

plt.subplot (2,4,5)
sns.distplot (df ['Other Debt'], hist = False)

plt.subplot (2,4,6)
sns.distplot (df ['DebtIncomeRatio'], hist = False)

plt.subplot (2,4,7)
df.Edu.value_counts ().plot.bar ()

plt.subplot (2,4,8)
df.Defaulted.value_counts ().plot.bar ()

plt.tight_layout ()


sns.pairplot (df [['Age', 'Edu', 'Years Employed', 'Income', 'Card Debt',
       'Other Debt', 'DebtIncomeRatio']])
plt.figure (figsize = (10,7))
sns.heatmap (df [['Age', 'Years Employed', 'Income', 'Card Debt',
       'Other Debt', 'DebtIncomeRatio']].corr (), annot = True, cmap = 'viridis')
df.isnull ().sum ()
df ['Defaulted'] = df.Defaulted.fillna (0)
df.head ()
from sklearn.preprocessing import StandardScaler
X = df.values[:,1:]

Clus_dataSet = StandardScaler().fit_transform(X)

Clus_dataSet [0:3]
inertia = {}
for k in range (2,15):
    kmeans = KMeans(init = "k-means++", n_clusters = k, n_init = 12, random_state = 42, algorithm='elkan').fit(Clus_dataSet)
    inertia.update ({k : kmeans.inertia_})

inertia_df = pd.DataFrame (inertia, index = [0]).transpose ()
inertia_df.columns = ['inertia']

inertia_df.plot ()
k_means = KMeans(init = "k-means++", n_clusters = 3, n_init = 12, random_state = 42, algorithm='elkan').fit(Clus_dataSet)
labels = k_means.labels_
labels [0:10]
centers = k_means.cluster_centers_
centers
df["Clus_km"] = labels
round (df.iloc [:,1:].groupby('Clus_km').mean())
area = np.pi * ( X[:, 1])**2  
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)

plt.show()
from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(5, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(np.float))
