#for mathematical operations
import numpy as np
import pandas as pd

#for visualisation
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

#for data preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

#k-means modelling
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# for Hierarchical clustering
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree
data = pd.read_csv("../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")
data.head()
data.tail()
# Shape of the dataset
data.shape
print("There are {} rows and {} columns in the dataset".format(data.shape[0],data.shape[1]))
data.columns
data.describe()
data.describe(include='object')
plt.rcParams['figure.figsize'] = (18,6)

#Creating distribution for all the numerical variables at a single place using subplots
plt.subplot(1,3,1)
sns.distplot(data['Age'],color = 'red')
plt.title("Distribution plot of Age", fontsize=16)

plt.subplot(1,3,2)
sns.distplot(data['Annual Income (k$)'], color='blue')
plt.title("Distribution plot of Annual Income (k$)", fontsize=16)

plt.subplot(1,3,3)
sns.distplot(data['Spending Score (1-100)'], color='brown')
plt.title("Distribution plot of Spending Score (1-100)", fontsize=16)
plt.rcParams['figure.figsize'] = (16,5)

#Countplot of Age Variable
sns.countplot(data['Age'], color='red')
plt.title("CountPlot of Age", fontsize=16)
plt.rcParams['figure.figsize'] = (16,5)

#Countplot of Annual Income
sns.countplot(data['Annual Income (k$)'], color='brown')
plt.title("CountPlot of Annual Income", fontsize=16)

plt.xticks(rotation=90)
plt.show()
plt.rcParams['figure.figsize'] = (16,5)

#Countplot of Spending Score
sns.countplot(data['Spending Score (1-100)'], color='blue')
plt.title("CountPlot of Spending Score", fontsize=16)

plt.xticks(rotation=90)
plt.show()
plt.rcParams['figure.figsize'] = (18,6)

plt.subplot(1,3,1)
sns.boxplot(data['Age'],color = 'red')
plt.title("Distribution plot of Age", fontsize=16)

plt.subplot(1,3,2)
sns.boxplot(data['Annual Income (k$)'], color='blue')
plt.title("Distribution plot of Annual Income (k$)", fontsize=16)

plt.subplot(1,3,3)
sns.boxplot(data['Spending Score (1-100)'], color='brown')
plt.title("Distribution plot of Spending Score (1-100)", fontsize=16)
plt.rcParams['figure.figsize'] = (12,6)

plt.pie(data['Gender'].value_counts(), labels = ['Female','Male'],autopct = '%.2f%%')
plt.show()
data.groupby('Gender')['Spending Score (1-100)'].mean()
sns.scatterplot(x='Spending Score (1-100)', y='Annual Income (k$)', data=data)
plt.style.use('fivethirtyeight')
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)',hue='Age', data=data)
sns.barplot(data['Gender'], data['Annual Income (k$)'])
sns.barplot(data['Gender'], data['Spending Score (1-100)'])
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
data.head()
sc = StandardScaler()

data_sc = sc.fit_transform(data.drop(['CustomerID','Gender'], axis=1))
data_sc_df = pd.DataFrame(data_sc)
data_sc_df.columns = ['Age','Annual Income (k$)','Spending Score (1-100)']
data_ca = data[['CustomerID','Gender']]
data_sc_new = pd.concat([data_ca,data_sc_df], axis=1)
data_sc_new.head()
data_k = data_sc_new[['Age','Annual Income (k$)','Spending Score (1-100)']]
data_k.head()
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
import numpy as np
from math import isnan
 
def hopkins(X):
    d = X.shape[1]
    #d = len(vars) # columns
    n = len(X) # rows
    m = int(0.1 * n) 
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
 
    rand_X = sample(range(0, n, 1), m)
 
    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])
 
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print(ujd, wjd)
        H = 0
 
    return H
hopkins(data_k)
# elbow-curve/SSD
ssd = []
clusters = [2, 3, 4, 5, 6, 7, 8]
for num_clusters in clusters:
    kmeans = KMeans(n_clusters=num_clusters, max_iter=200)
    kmeans.fit(data_k)
    
    ssd.append(kmeans.inertia_)
    
# plot the SSDs for each n_clusters
# ssd
plt.plot(ssd)
# silhouette analysis
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for num_clusters in range_n_clusters:
    
    # intialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(data_k)
    
    cluster_labels = kmeans.labels_
    
    # silhouette score
    silhouette_avg = silhouette_score(data_k, cluster_labels)
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))
    
    
kmeans = KMeans(n_clusters=5, max_iter=200)
kmeans.fit(data_k)
kmeans.labels_
data['Cust_ID']  = kmeans.labels_
data.head()

sns.boxplot(x='Cust_ID', y='Age', data=data)
sns.boxplot(x='Cust_ID', y='Annual Income (k$)', data=data)
sns.boxplot(x='Cust_ID', y='Spending Score (1-100)', data=data)
link = linkage(data_k, method='complete')
dendrogram(link)
plt.show()
clusters = cut_tree(link, n_clusters=5)
clusters.reshape(-1,)
data['Cust_ID_Hierarchy']  = clusters
data.head()
sns.boxplot(x='Cust_ID_Hierarchy', y='Spending Score (1-100)', data=data)
sns.boxplot(x='Cust_ID_Hierarchy', y='Annual Income (k$)', data=data)
sns.boxplot(x='Cust_ID_Hierarchy', y='Age', data=data)