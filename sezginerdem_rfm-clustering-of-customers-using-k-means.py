# Importing necessary libraries we are going to use



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import datetime as dt

import sklearn

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

from sklearn.preprocessing import MinMaxScaler

from scipy.cluster.hierarchy import linkage

from scipy.cluster.hierarchy import dendrogram

from scipy.cluster.hierarchy import cut_tree

!pip install yellowbrick

from yellowbrick.cluster import KElbowVisualizer

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 

warnings.filterwarnings("ignore", category=FutureWarning)



# to display all columns and rows and string formatting:



pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', None);

pd.set_option('display.float_format', lambda x: '%.0f' % x)

import matplotlib.pyplot as plt



#Data set reading process was performed.

df_2010_2011 = pd.read_excel("../input/online-retail-ii-data-set-from-ml-repository/online_retail_II.xlsx", sheet_name = "Year 2010-2011")

df = df_2010_2011.copy()
# We extract the returned transactions from the data.

returned = df[df["Invoice"].str.contains("C",na=False)].index

df = df.drop(returned, axis = 0)



# How much money has been earned per invoice? (It is necessary to create a new variable by multiplying two variables)

df["TotalPrice"] = df["Quantity"]*df["Price"]



# Missing observations were deleted.

df.dropna(inplace = True)
# try to determine the earliest invoice date

df["InvoiceDate"].min()



# try to determine the latest invoice date

df["InvoiceDate"].max()



# we set the latest date of dataset as todays`s date

import datetime as dt

today_date = dt.datetime(2011,12,9) 



# try to determine the latest transaction date for each customer

df.groupby("Customer ID").agg({"InvoiceDate":"max"}).head()



# if we substarct the latest date of transaction for each customer from today`s date, we can get the recency 

temp_df = (today_date - df.groupby("Customer ID").agg({"InvoiceDate":"max"}))



# change the column name from 'InvoiceDate'to 'Recency'

temp_df.rename(columns={"InvoiceDate": "Recency"}, inplace = True)



# just take the days 

recency_df = temp_df["Recency"].apply(lambda x: x.days) 

recency_df.head()
#finding frequency value of each customer. Lists invoices and counts them

temp_df = df.groupby(["Customer ID","Invoice"]).agg({"Invoice":"count"})



#counts all invoices for each customer

freq_df = temp_df.groupby("Customer ID").agg({"Invoice":"count"})



#changes column name (Invoice -> Frequency)

freq_df.rename(columns={"Invoice": "Frequency"}, inplace = True)

freq_df.head() 
#shows each customer's total spendings

monetary_df = df.groupby("Customer ID").agg({"TotalPrice":"sum"})



#changes column name (TotalPrice -> Monetary)

monetary_df.rename(columns = {"TotalPrice": "Monetary"}, inplace = True)

monetary_df.head()
# Try to combine recency, frequency, and monetary as columns in rfm dataframe

rfm = pd.concat([recency_df, freq_df, monetary_df],  axis=1)

rfm.head()



# if interested, outliers can be determined for RFM scores

for feature in ["Recency","Frequency","Monetary"]:



    Q1 = rfm[feature].quantile(0.05)

    Q3 = rfm[feature].quantile(0.95)

    IQR = Q3-Q1

    upper = Q3 + 1.5*IQR

    lower = Q1 - 1.5*IQR



    if rfm[(rfm[feature] > upper) | (rfm[feature] < lower)].any(axis=None):

        print(feature,"yes")

        print(rfm[(rfm[feature] > upper) | (rfm[feature] < lower)].shape[0])

    else:

        print(feature, "no")

        

mms = MinMaxScaler((0,1))

cols = rfm.columns

index = rfm.index

scaled_rfm = mms.fit_transform(rfm)

scaled_rfm = pd.DataFrame(scaled_rfm, columns=cols, index = index)

scaled_rfm.head()
# Clustering with the K-Means Algorithm

sc = MinMaxScaler((0,1))

df = sc.fit_transform(rfm)

kmeans = KMeans(n_clusters = 10)

k_fit = kmeans.fit(df)

k_fit.labels_
#Determining the Optimum Number of Clusters

kmeans = KMeans(n_clusters = 10)

k_fit = kmeans.fit(df)

ssd = []



K = range(1,30)



for k in K:

    kmeans = KMeans(n_clusters = k).fit(df)

    ssd.append(kmeans.inertia_)



plt.plot(K, ssd, "bx-")

plt.xlabel("Distance Residual Sums Versus Different k Values")

plt.title("Elbow method for Optimum number of clusters")
from yellowbrick.cluster import KElbowVisualizer

kmeans = KMeans()

visu = KElbowVisualizer(kmeans, k = (2,20))

visu.fit(df)

visu.poof();
kmeans = KMeans(n_clusters = 6).fit(df)

kumeler = kmeans.labels_

pd.DataFrame({"Customer ID": rfm.index, "Kumeler": kumeler})

rfm["cluster_no"] = kumeler

rfm["cluster_no"] = rfm["cluster_no"] + 1

rfm.groupby("cluster_no").agg({"cluster_no":"count"})
rfm.head()
rfm.groupby("cluster_no").agg({"mean"})
from scipy.cluster.hierarchy import linkage

from scipy.cluster.hierarchy import dendrogram

hc_complete = linkage(rfm, "complete")

hc_average = linkage(rfm, "average")



plt.figure(figsize = (15,10))

plt.title("Hierarchical Cluster Dendrogram")

plt.xlabel("Observation Unit")

plt.ylabel("Distance")

dendrogram(hc_complete,

           truncate_mode = "lastp",

           p = 10,

           show_contracted = True,

          leaf_font_size = 10);
cluster_labels = cut_tree(hc_complete, n_clusters=4).reshape(-1, )

rfm['Cluster_Labels'] = cluster_labels

rfm['Cluster_Labels'] = rfm['Cluster_Labels'] + 1

rfm.groupby("Cluster_Labels").agg(np.mean)
sns.boxplot(x='Cluster_Labels', y='Monetary', data=rfm);
sns.boxplot(x='Cluster_Labels', y='Frequency', data=rfm);
sns.boxplot(x='Cluster_Labels', y='Recency', data=rfm);