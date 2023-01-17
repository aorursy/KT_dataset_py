import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import datetime as dt



import sklearn

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score



from scipy.cluster.hierarchy import linkage

from scipy.cluster.hierarchy import dendrogram

from scipy.cluster.hierarchy import cut_tree
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# read the dataset

retail_df = pd.read_csv(r"/kaggle/input/learning-data/OnlineRetail.csv", sep=",", encoding="ISO-8859-1", header=0)

retail_df.head()
# basics of the df

retail_df.info()
# missing values

round(100*(retail_df.isnull().sum())/len(retail_df), 2)
# drop all rows having missing values

retail_df = retail_df.dropna()

retail_df.shape
retail_df.head()
# new column: amount 

retail_df['amount'] = retail_df['Quantity']*retail_df['UnitPrice']

retail_df.head()
# monetary

grouped_df = retail_df.groupby('CustomerID')['amount'].sum()

grouped_df = grouped_df.reset_index()

grouped_df.head()
# frequency

frequency = retail_df.groupby('CustomerID')['InvoiceNo'].count()

frequency = frequency.reset_index()

frequency.columns = ['CustomerID', 'frequency']

frequency.head()
# merge the two dfs

grouped_df = pd.merge(grouped_df, frequency, on='CustomerID', how='inner')

grouped_df.head()
retail_df.head()
# recency

# convert to datetime

retail_df['InvoiceDate'] = pd.to_datetime(retail_df['InvoiceDate'], 

                                          format='%d-%m-%Y %H:%M')
retail_df.head()
# compute the max date

max_date = max(retail_df['InvoiceDate'])

max_date
# compute the diff

retail_df['diff'] = max_date - retail_df['InvoiceDate']

retail_df.head()
# recency

last_purchase = retail_df.groupby('CustomerID')['diff'].min()

last_purchase = last_purchase.reset_index()

last_purchase.head()
# merge

grouped_df = pd.merge(grouped_df, last_purchase, on='CustomerID', how='inner')

grouped_df.columns = ['CustomerID', 'amount', 'frequency', 'recency']

grouped_df.head()
# number of days only

grouped_df['recency'] = grouped_df['recency'].dt.days

grouped_df.head()
# 1. outlier treatment

fig=plt.figure(figsize=(20,8))

plt.subplot(1,2,1)

sns.violinplot(y=grouped_df['recency'])

plt.subplot(1,2,2)

sns.boxplot(y=grouped_df['recency'])

plt.show()
# two types of outliers:

# - statistical

# - domain specific
# removing (statistical) outliers

Q1 = grouped_df.amount.quantile(0.05)

Q3 = grouped_df.amount.quantile(0.95)

IQR = Q3 - Q1

grouped_df = grouped_df[(grouped_df.amount >= Q1 - 1.5*IQR) & (grouped_df.amount <= Q3 + 1.5*IQR)]



# outlier treatment for recency

Q1 = grouped_df.recency.quantile(0.05)

Q3 = grouped_df.recency.quantile(0.95)

IQR = Q3 - Q1

grouped_df = grouped_df[(grouped_df.recency >= Q1 - 1.5*IQR) & (grouped_df.recency <= Q3 + 1.5*IQR)]



# outlier treatment for frequency

Q1 = grouped_df.frequency.quantile(0.05)

Q3 = grouped_df.frequency.quantile(0.95)

IQR = Q3 - Q1

grouped_df = grouped_df[(grouped_df.frequency >= Q1 - 1.5*IQR) & (grouped_df.frequency <= Q3 + 1.5*IQR)]



# 2. rescaling

rfm_df = grouped_df[['amount', 'frequency', 'recency']]



# instantiate

scaler = StandardScaler()



# fit_transform

rfm_df_scaled = scaler.fit_transform(rfm_df)

rfm_df_scaled.shape
rfm_df_scaled = pd.DataFrame(rfm_df_scaled)

rfm_df_scaled.columns = ['amount', 'frequency', 'recency']

rfm_df_scaled.head()
# k-means with some arbitrary k

kmeans = KMeans(n_clusters=4, max_iter=50)

kmeans.fit(rfm_df_scaled)
kmeans.labels_
# help(KMeans)
# elbow-curve/SSD

fig=plt.figure(figsize=(20,8))

ssd = []

range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for num_clusters in range_n_clusters:

    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)

    kmeans.fit(rfm_df_scaled)

    

    ssd.append(kmeans.inertia_)

    

# plot the SSDs for each n_clusters

# ssd

plt.plot(ssd);
# silhouette analysis

range_n_clusters = [2, 3, 4, 5, 6, 7, 8]



for num_clusters in range_n_clusters:

    

    # intialise kmeans

    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)

    kmeans.fit(rfm_df_scaled)

    

    cluster_labels = kmeans.labels_

    

    # silhouette score

    silhouette_avg = silhouette_score(rfm_df_scaled, cluster_labels)

    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))

    

    
from sklearn.neighbors import NearestNeighbors

from random import sample

from numpy.random import uniform

import numpy as np

from math import isnan

import pandas as pd

 

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
#Use the Hopkins Statistic function by passing the above dataframe as a paramter

hopkins(rfm_df_scaled)
# final model with k=3

kmeans = KMeans(n_clusters=3, max_iter=50)

kmeans.fit(rfm_df_scaled)
kmeans.labels_
# assign the label

grouped_df['cluster_id'] = kmeans.labels_

grouped_df.head()