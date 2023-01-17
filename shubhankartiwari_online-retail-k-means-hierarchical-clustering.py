import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime as dt

import os

import sklearn

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import linkage

from scipy.cluster.hierarchy import dendrogram

from scipy.cluster.hierarchy import cut_tree
retail = pd.read_csv('../input/online-retail-customer-clustering/OnlineRetail.csv', sep=",", encoding="ISO-8859-1", header=0)

retail.head()

retail.shape

retail.info()

retail.describe()

df_null = round(100*(retail.isnull().sum())/len(retail), 2)

df_null
retail = retail.dropna() 

retail.shape

retail['CustomerID'] = retail['CustomerID'].astype(str)

retail['Amount'] = retail['Quantity']*retail['UnitPrice']

rfm_m = retail.groupby('CustomerID')['Amount'].sum()

rfm_m = rfm_m.reset_index()

rfm_m.head()
rfm_f = retail.groupby('CustomerID')['InvoiceNo'].count()

rfm_f = rfm_f.reset_index()

rfm_f.columns = ['CustomerID', 'Frequency']

rfm_f.head()

rfm = pd.merge(rfm_m, rfm_f, on='CustomerID', how='inner')

rfm.head()

retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'],format='%d-%m-%Y %H:%M')

max_date = max(retail['InvoiceDate'])

max_date

retail['Diff'] = max_date - retail['InvoiceDate']

retail.head()
rfm_p = retail.groupby('CustomerID')['Diff'].min()

rfm_p = rfm_p.reset_index()

rfm_p.head()

rfm_p['Diff'] = rfm_p['Diff'].dt.days

rfm_p.head()

rfm = pd.merge(rfm, rfm_p, on='CustomerID', how='inner')

rfm.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']

rfm.head()

attributes = ['Amount','Frequency','Recency']

plt.rcParams['figure.figsize'] = [10,8]

sns.boxplot(data = rfm[attributes], orient="v", palette="Set2" ,whis=1.5,saturation=1, width=0.7)

plt.title("Outliers Variable Distribution", fontsize = 14, fontweight = 'bold')

plt.ylabel("Range", fontweight = 'bold')

plt.xlabel("Attributes", fontweight = 'bold')
Q1 = rfm.Amount.quantile(0.05)

Q3 = rfm.Amount.quantile(0.95)

IQR = Q3 - Q1

rfm = rfm[(rfm.Amount >= Q1 - 1.5*IQR) & (rfm.Amount <= Q3 + 1.5*IQR)]

Q1 = rfm.Recency.quantile(0.05)

Q3 = rfm.Recency.quantile(0.95)

IQR = Q3 - Q1

rfm = rfm[(rfm.Recency >= Q1 - 1.5*IQR) & (rfm.Recency <= Q3 + 1.5*IQR)]

Q1 = rfm.Frequency.quantile(0.05)

Q3 = rfm.Frequency.quantile(0.95)

IQR = Q3 - Q1

rfm = rfm[(rfm.Frequency >= Q1 - 1.5*IQR) & (rfm.Frequency <= Q3 + 1.5*IQR)]

rfm_df = rfm[['Amount', 'Frequency', 'Recency']]
scaler = StandardScaler()

rfm_df_scaled = scaler.fit_transform(rfm_df)

rfm_df_scaled.shape

rfm_df_scaled = pd.DataFrame(rfm_df_scaled)

rfm_df_scaled.columns = ['Amount', 'Frequency', 'Recency']

rfm_df_scaled.head()

kmeans = KMeans(n_clusters=4, max_iter=50)

kmeans.fit(rfm_df_scaled)

kmeans.labels_

ssd = []

range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for num_clusters in range_n_clusters:

    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)

    kmeans.fit(rfm_df_scaled)

    ssd.append(kmeans.inertia_)

plt.plot(ssd)
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for num_clusters in range_n_clusters:

    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)

    kmeans.fit(rfm_df_scaled)

    cluster_labels = kmeans.labels_

    silhouette_avg = silhouette_score(rfm_df_scaled, cluster_labels)

    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))

kmeans = KMeans(n_clusters=3, max_iter=50)

kmeans.fit(rfm_df_scaled)

kmeans.labels_

rfm['Cluster_Id'] = kmeans.labels_

rfm.head()

sns.boxplot(x='Cluster_Id', y='Amount', data=rfm)
sns.boxplot(x='Cluster_Id', y='Frequency', data=rfm)

sns.boxplot(x='Cluster_Id', y='Recency', data=rfm)
mergings = linkage(rfm_df_scaled, method="single", metric='euclidean')

dendrogram(mergings)

plt.show()



mergings = linkage(rfm_df_scaled, method="complete", metric='euclidean')

dendrogram(mergings)

plt.show()
mergings = linkage(rfm_df_scaled, method="average", metric='euclidean')

dendrogram(mergings)

plt.show()
cluster_labels = cut_tree(mergings, n_clusters=3).reshape(-1, )

cluster_labels

rfm['Cluster_Labels'] = cluster_labels

rfm.head()

sns.boxplot(x='Cluster_Labels', y='Amount', data=rfm)
sns.boxplot(x='Cluster_Labels', y='Frequency', data=rfm)
sns.boxplot(x='Cluster_Labels', y='Recency', data=rfm)