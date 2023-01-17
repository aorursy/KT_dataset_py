import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set()



from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_samples, silhouette_score



from sklearn.decomposition import PCA

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import MinMaxScaler,StandardScaler
# Load online retail transactional data set

ORC = pd.read_excel('../input/online-retail-data-set-from-uci-ml-repo/Online Retail.xlsx')

print("Retail transactions in the data set : {}".format(ORC.shape[0]))
# First five transactions

ORC.head()
# Summary of data set.

ORC.info()
print("Duplicate transactions : {}".format(ORC.duplicated().sum()))
# First five duplicate transactions

ORC[ORC.duplicated()].head()
# Select transactions with InvoiceNo = '536409' & StockCode = '21866'

ORC.loc[(ORC['InvoiceNo'].astype(str) == '536409') & (ORC['StockCode'].astype(str) == '21866')]
# Remove duplicates

ORC = ORC.drop(index=ORC[ORC.duplicated()].index)

print("Retail transactions after duplicates removal : {}".format(ORC.shape[0]))
# Detect missing values

ORC.isnull().sum()
# Descriptive statistics for StockCode.

ORC['StockCode'].describe()
# Descriptive statistics for Quantity.

ORC['Quantity'].describe()
# Inspect min and max Quantity transactions

ORC.loc[(ORC['Quantity']==-80995) | (ORC['Quantity']==80995)]
# Cancelled/Reversed invoices

print("Cancelled invoices/transactions : {}".format(ORC[ORC['InvoiceNo'].astype(str).str[0] == 'C'].shape[0]))
# Descriptive statistics for UnitPrice

ORC['UnitPrice'].describe()
# Display transactions with negative unit price.

ORC[ORC['UnitPrice'] < 0]
# Dsiplay transaction with unit price > 10000

ORC[ORC['UnitPrice'] > 10000]
print("Transactions with zero unit price : {}".format(ORC[ORC['UnitPrice'] == 0].shape[0]))
#First five transactions with zero unit price

ORC[ORC['UnitPrice'] == 0].head()
# top 10 selling products by their counts in data set.

ORC['Description'].value_counts().sort_values(ascending=False)[:10]
# Descriptive Statistics for Country

ORC['Country'].describe()
#Remove transactions with missing customer ids

ORC = ORC.drop(index=ORC[ORC['CustomerID'].isnull()].index)

print("Retail transactions after removing missing customer ids  : {}".format(ORC.shape[0]))
import datetime as dt



ORC['InvoiceDate'] = pd.to_datetime(ORC['InvoiceDate'])

#ORC['SalesAmount'] = ORC['Quantity'] * ORC['UnitPrice']

ORC['Month'] = ORC['InvoiceDate'].dt.month

ORC['Day'] = ORC['InvoiceDate'].dt.day

ORC['Hour'] = ORC['InvoiceDate'].dt.hour
hourly_sales = ORC[['Hour', 'Quantity']].groupby('Hour').sum()

#hourly_sales.plot(kind='bar')

plt.figure(figsize=(8,6))

plt.title("Hourly Sales", fontsize=14)

sns.barplot(hourly_sales.index, hourly_sales['Quantity'])
daily_sales = ORC[['Day', 'Quantity']].groupby('Day').sum()

plt.figure(figsize=(10,8))

plt.title("Daily Sales", fontsize=14)

sns.barplot(daily_sales.index, daily_sales['Quantity'])
monthly_sales = ORC[['Month', 'Quantity']].groupby('Month').sum()

plt.figure(figsize=(8,6))

plt.title("Monthly Sales", fontsize=14)

sns.barplot(monthly_sales.index, monthly_sales['Quantity'])
# Recency -> The freshness of customer purchase



# Calculate latest date from data set.

max_date = ORC['InvoiceDate'].max()

# Calculate days passed since customer's last purchase.

ORC['Days_passed'] = max_date -  ORC['InvoiceDate']

ORC['Days_passed'] = ORC['Days_passed'].dt.days

# Group Recency by customer id

recency = ORC[['CustomerID', 'Days_passed']].groupby('CustomerID').min()

recency.head(5)
# Frequency of the customer transactions

frequency = ORC[['CustomerID','InvoiceNo']].groupby('CustomerID').count()

frequency.head()
# Monetory -> purchasing power of the customer

ORC['SaleAmount'] = ORC['Quantity'] * ORC['UnitPrice']

monetory = ORC[['CustomerID', 'SaleAmount']].groupby('CustomerID').sum()

monetory.head()
# Merge recency, frequency and monetory dataframes

RFM = recency.merge(frequency,on='CustomerID').merge(monetory, on='CustomerID')

RFM = RFM.rename(columns={"Days_passed": "Recency", "InvoiceNo": "Frequency", "SaleAmount" : "Monetory"})

RFM.head()
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]



for n_clusters in range_n_clusters:

    clusterer = KMeans(n_clusters=n_clusters, random_state=5)

    cluster_labels = clusterer.fit_predict(RFM)

    

    silhouette_avg = silhouette_score(RFM, cluster_labels)

    print("for n_clusters =", n_clusters, "Average Silhouette score = ", silhouette_avg)
# Kmeans with number of clusers = 4

clusterer = KMeans(n_clusters=4, random_state=5)

cluster_labels = clusterer.fit_predict(RFM)
RFM['Cluster'] = cluster_labels

RFM.groupby('Cluster').mean()
# Reduce dimension to 2 with PCA

pca = make_pipeline(StandardScaler(),

                    PCA(n_components=2, random_state=43))

RFM_transformed = pca.fit_transform(RFM)
plt.figure(figsize=(10,8))

sns.scatterplot(RFM_transformed[:,0], RFM_transformed[:,1], hue=cluster_labels)

plt.xlabel('1st Principle Component')

plt.ylabel('2nd Principle Component')

plt.title("Clusters", fontsize=14)