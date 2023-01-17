# import required libraries for dataframe and visualization



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime as dt



# import required libraries for clustering

import sklearn

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score
# Read data

data = pd.read_csv('../input/online-retail-customer-clustering/OnlineRetail.csv',encoding="ISO-8859-1")

print(data.head())
# shape

print(data.shape)
# data description

print(data.describe())
# Calculate missing values % in original data

data_null = round(100 * (data.isnull().sum()) / len(data), 2)

print(data_null)
# Drop rows with missing values

data = data.dropna()

print(data.shape)
# Change data type of Customer Id; they are not numeric in essence

data['CustomerID'] = data['CustomerID'].astype(str)
# New Attribute : Recency

# Reformat datetime

data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], format='%d-%m-%Y %H:%M')

print(data['InvoiceDate'])
# Compute the difference between most recent date and transaction date

data['Diff'] = max(data['InvoiceDate']) - data['InvoiceDate']

print(data.head())
# Compute last transaction date to get the recency of customers

rfm_r = data.groupby('CustomerID', as_index=False)['Diff'].min()
# Extract number of days only

rfm_r['Diff'] = rfm_r['Diff'].dt.days

rfm_r.columns = ['CustomerID', 'Recency']

print(rfm_r.head())
# New Attribute : Frequency

rfm_f = data.groupby('CustomerID', as_index=False)['InvoiceNo'].count()

rfm_f.columns = ['CustomerID', 'Frequency']

print(rfm_f.head())
# New Attribute : Monetary

data['Amount'] = data['Quantity'] * data['UnitPrice']

rfm_m = data.groupby('CustomerID', as_index=False)['Amount'].sum()

rfm_m.columns = ['CustomerID', 'Amount']

print(rfm_m.head())
# Combine R, F, M

rfm = pd.concat((rfm_r['Recency'], rfm_f['Frequency'], rfm_m['Amount']), axis=1)

print(rfm.head())
# Remove negative amounts (excluding goods refund)

rfm = rfm[rfm.Amount > 0]
# Standardize data

from sklearn.preprocessing import StandardScaler



# Store original column names and data

keys = rfm.keys()

rfm_unscaled = rfm



# Scale rfm

scaler = StandardScaler()

rfm = scaler.fit_transform(rfm)

rfm = pd.DataFrame(rfm, columns=keys)

print(rfm.head())
# Remove values outside mean +/- 3 std range

for key in rfm.keys():

    mean = np.mean(rfm[key])

    std = np.std(rfm[key])

    rfm = rfm[np.abs(rfm[key] - mean) / std <= 3]
# Draw elbow curve to find optimal K value

# 2 <= i <= 10

inertia = {}

for i in range(2, 11):

    kmeans = KMeans(n_clusters=i, max_iter=1000)

    kmeans.fit(rfm)

    inertia[i] = kmeans.inertia_



for k, v in inertia.items():

    print(str(k), ': ', str(v))
# Plot for each K value

plt.subplots()

plt.plot(list(inertia.values()), 'b+-')

plt.show()
# # Silhouette analysis

for num_clusters in range(2,10):

    kmeans = KMeans(n_clusters=num_clusters, max_iter=1000)

    kmeans.fit(rfm)

    cluster_labels = kmeans.labels_

    

    silhouette_avg = silhouette_score(rfm, cluster_labels)

    print("For n_clusters={0}, the silhouette score is {1:.4f}".format(num_clusters, silhouette_avg))
# Final model with k=3

kmeans = KMeans(n_clusters=3, max_iter=1000)

kmeans.fit(rfm)

print(kmeans.labels_)
# Assign label

rfm['Cluster_Id'] = kmeans.labels_

print(rfm.head())

print(rfm.shape)
# Scatter plot

plt.subplots()

plt.scatter(x=rfm['Recency'], y=rfm['Amount'], c=rfm['Cluster_Id'], alpha=0.4)

plt.xlabel('Recency')

plt.ylabel('Amount')

plt.title("Clustering: Recency vs Amount")



plt.subplots()

plt.scatter(x=rfm['Frequency'], y=rfm['Amount'], c=rfm['Cluster_Id'], alpha=0.4)

plt.xlabel('Frequency')

plt.ylabel('Amount')

plt.title("Clustering: Frequency vs Amount")



plt.subplots()

plt.scatter(x=rfm['Frequency'], y=rfm['Recency'], c=rfm['Cluster_Id'], alpha=0.4)

plt.xlabel('Frequency')

plt.ylabel('Recency')

plt.title("Clustering: Frequency vs Recency")
# Create 3D scatter plot

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(xs=rfm['Recency'], ys=rfm['Frequency'], zs=rfm['Amount'], c=rfm['Cluster_Id'])



plt.title('RFM Clustering')

ax.set_xlabel('Recency')

ax.set_ylabel('Frequency')

ax.set_zlabel('Amount')
# Restore rfm

rfm = rfm_unscaled

print(rfm.head())



# Remove values outside mean +/- 3 std range

for key in rfm.keys():

    mean = np.mean(rfm[key])

    std = np.std(rfm[key])

    rfm = rfm[np.abs(rfm[key] - mean) / std <= 3]

# Refit model with k=3 to unscaled rfm

kmeans = KMeans(n_clusters=3, max_iter=1000)

kmeans.fit(rfm)

print(kmeans.labels_)
# Assign label

rfm['Cluster_Id'] = kmeans.labels_

print(rfm.head())

print(rfm.shape)
# Create a new data frame for cluster analysis

cluster_sale = rfm.groupby('Cluster_Id', as_index=False)['Amount'].sum()

cluster_sale.columns = ['ClusterId', 'Revenue']
# Count the number of customers by their cluster id

cluster_sale['CustomerCount'] = rfm['Cluster_Id'].value_counts()

cluster_sale['Customer%'] = cluster_sale['CustomerCount'] / sum(cluster_sale['CustomerCount'])
# Calculate the average recency of customers

cluster_sale['MeanRecency'] = rfm.groupby('Cluster_Id', as_index=False)['Recency'].mean()['Recency']
# Calculate the number of transactions done by clusters

cluster_sale['TransactionCount'] = rfm.groupby('Cluster_Id', as_index=False)['Frequency'].sum()['Frequency']

cluster_sale['Transaction%'] = cluster_sale['TransactionCount'] / sum(cluster_sale['TransactionCount'])
# Calculate the average number of times customers do shopping

cluster_sale['MeanFrequency'] = cluster_sale['TransactionCount'] / cluster_sale['CustomerCount']
# Calculate the percentage of total revenue by clusters

cluster_sale['Revenue%'] = cluster_sale['Revenue'] / sum(cluster_sale['Revenue'])
# Calculate average revenue per transaction

cluster_sale['ARPT'] = cluster_sale['Revenue'] / cluster_sale['TransactionCount']
# Calculate average revenue per customer

cluster_sale['ARPC'] = cluster_sale['Revenue'] / cluster_sale['CustomerCount']
# Reorganize columns for easier reading

cluster_sale = cluster_sale[

    ['ClusterId', 'CustomerCount', 'Customer%', 'MeanRecency', 'MeanFrequency', 'Revenue', 'Revenue%',

     'TransactionCount', 'Transaction%', 'ARPT', 'ARPC']]
print(cluster_sale.head())