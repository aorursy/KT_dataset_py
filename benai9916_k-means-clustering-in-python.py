import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.cluster import KMeans
seller = pd.read_csv('/kaggle/input/facebook-live-sellers-in-thailand-uci-ml-repo/Live.csv')
# first five row
seller.head()
# size of datset
seller.shape
# statistical summary of numerical variables
seller.describe()
# summary about dataset
seller.info()
# check for missing values
seller.isna().sum() 
seller = seller.dropna(axis=1)
# we have drop 4 columns
seller.isna().sum() 
# check the unique values

print(seller['status_id'].unique())
# check the number of unique values

print(seller['status_id'].nunique())
# check the unique values

seller['status_published'].unique()
# check the number of unique values

print(seller['status_published'].nunique())
# check the unique values

seller['status_type'].unique()
# check the number of unique values
seller['status_type'].nunique
seller.drop(['status_id', 'status_published'], axis=1, inplace=True)
# check the summary
seller.info()
sns.pairplot(seller)
X = seller

y = seller['status_type']
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

X['status_type'] = le.fit_transform(X['status_type'])

y = le.transform(y)
# check x
X.head()
from sklearn.preprocessing import MinMaxScaler

min_scal = MinMaxScaler()

X = min_scal.fit_transform(X)
X = pd.DataFrame(X, columns=[seller.columns])
X.head()
k_means = KMeans(n_clusters=2, random_state=42) 

k_means.fit(X)
# model parameter study
k_means.cluster_centers_
# calculate model inertia

k_means.inertia_
from sklearn.metrics import silhouette_score

# Silhouette analysis
range_n_clusters = [2, 3, 4, 5, 6]

for num_clusters in range_n_clusters:
    
    # intialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(X)
    
    cluster_labels = kmeans.labels_
    
    # silhouette score
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))
labels = k_means.labels_

# check how many of the samples were correctly labeled
correct_labels = sum(y == labels)

print("Result: {} out of {} samples were correctly labeled.".format(correct_labels, y.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
cs = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 7)
    
    kmeans.fit(X)
    
    cs.append(kmeans.inertia_)

# plot the 
plt.plot(range(1, 11), cs)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()
# K-Means model with 3 clusters

k_means3 = KMeans(n_clusters=3,max_iter = 400, n_init = 10, random_state=7)

k_means3.fit(X)

# check how many of the samples were correctly labeled
labels = k_means3.labels_

correct_labels3 = sum(y == labels)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels3, y.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels3/float(y.size)))
# Inertia
print("Inertia: ",k_means3.inertia_)

# silhouette score
silhouette_avg3 = silhouette_score(X, (k_means3.labels_))
print("The silhouette score is ", silhouette_avg3)
# K-Means model with 4 clusters

k_means4 = KMeans(n_clusters=4, max_iter = 400, n_init = 10, random_state=7)

k_means4.fit(X)

# check how many of the samples were correctly labeled
labels = k_means4.labels_

correct_labels4 = sum(y == labels)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels4, y.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels4/float(y.size)))
# Inertia
print("Inertia: ",k_means4.inertia_)

# silhouette score
silhouette_avg4 = silhouette_score(X, (k_means4.labels_))
print("The silhouette score is ", silhouette_avg4)
