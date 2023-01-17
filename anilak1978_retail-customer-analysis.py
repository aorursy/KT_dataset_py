# Import Standard packages

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
# Load the available data and overview

df=pd.read_csv("https://raw.githubusercontent.com/anilak1978/ecommerce/master/OnlineRetail.csv", encoding="ISO-8859-1")

df.head()
# Look for missing values

df.isnull().sum()
# Look for datatypes

df.dtypes
# Add Revenue variable and update InvoiceDate 

df["Revenue"]=df["UnitPrice"]*df["Quantity"]

df["InvoiceDate"]=pd.to_datetime(df["InvoiceDate"]).dt.date

df["InvoiceMonth"]=pd.DatetimeIndex(df["InvoiceDate"]).month

df["InvoiceYear"]=pd.DatetimeIndex(df["InvoiceDate"]).year

df.head()
# basic statistical analysis

df.info()
df.describe()
#Monthly Revenue Overview

df_revenue=df.groupby(["InvoiceMonth", "InvoiceYear"])["Revenue"].sum().reset_index()

plt.figure(figsize=(15,10))

sns.barplot(x="InvoiceMonth", y="Revenue", hue="InvoiceYear", data=df_revenue)

plt.title("Monthly Revenue")

plt.xlabel("Month")

plt.ylabel("Revenue")
# Monthly Revenue Overview

plt.figure()

sns.relplot(x="InvoiceMonth", y="Revenue", hue="InvoiceYear", kind="line", data=df_revenue, height=10, aspect=15/10)

plt.title("Monthly Revenue")

plt.xlabel("Month")

plt.ylabel("Revenue")
# Look at the December 2011 data

df_december_2011=df.query("InvoiceMonth==12 and InvoiceYear==2011")

df_december_2011
# Monthly Items Sold Overview

df_quantity=df.groupby(["InvoiceMonth", "InvoiceYear"])["Quantity"].sum().reset_index()

plt.figure(figsize=(15,10))

sns.barplot(x="InvoiceMonth", y="Quantity", data=df_quantity)

plt.title("Monthly Items Sold")

plt.xlabel("Month")

plt.ylabel("Items Sold")
# Monthly Active Customers

df_active=df.groupby(["InvoiceMonth", "InvoiceYear"])["CustomerID"].nunique().reset_index()

plt.figure(figsize=(15,10))

sns.barplot(x="InvoiceMonth", y="CustomerID", hue="InvoiceYear", data=df_active)

plt.title("Monthly Active Users")

plt.xlabel("Month")

plt.ylabel("Active Users")
# Average Revenue per Month

df_revenue_avg=df.groupby(["InvoiceMonth", "InvoiceYear"])["Revenue"].mean().reset_index()

plt.figure(figsize=(15,10))

sns.barplot(x="InvoiceMonth", y="Revenue", data=df_revenue)

plt.title("Monthly Average Revenue ")

plt.xlabel("Month")

plt.ylabel("Revenue")
# New vs Existing Users

df_first_purchase=df.groupby(["CustomerID"])["InvoiceDate"].min().reset_index()

df_first_purchase.columns=["CustomerID", "FirstPurchaseDate"]

df=pd.merge(df, df_first_purchase, on="CustomerID")

df["UserType"]="New"

df.loc[df["InvoiceDate"]>df["FirstPurchaseDate"], "UserType"]="Existing"

df.head()
# New vs Existing User Revenue Analysis

df_new_revenue=df.groupby(["InvoiceMonth", "InvoiceYear", "UserType"])["Revenue"].sum().reset_index()

plt.figure()

sns.relplot(x="InvoiceMonth", y="Revenue", hue="UserType", data=df_new_revenue, kind="line", height=12, aspect=18/10)

plt.title("New vs Existing Customer Revenue Overview")

plt.xlabel("Month")

plt.ylabel("Revenue")
# Recency Calculation

df_user=pd.DataFrame(df["CustomerID"].unique())

df_user.columns=["CustomerID"]

df_last_purchase=df.groupby(["CustomerID"])["InvoiceDate"].max().reset_index()

df_last_purchase.columns=["CustomerID", "LastPurchaseDate"]

df_last_purchase["Recency"]=(df_last_purchase["LastPurchaseDate"].max()-df_last_purchase["LastPurchaseDate"]).dt.days

df_recency=pd.merge(df_user, df_last_purchase[["CustomerID", "Recency"]])

df_recency.head()
# Look at the distribution of Recency

plt.figure(figsize=(15,10))

sns.distplot(df_recency["Recency"])

plt.title("Recency Distribution Within the Customers")

plt.xlabel("Recency")

plt.ylabel("Customer Count")
# use KMeans Clustering for Recency Clustering

from sklearn.cluster import KMeans

# find out how many clusters are optimal

y=df_recency[["Recency"]] # label what we are clustering

dic={} # store the clustering values in a dictionary

for k in range(1,10):

    kmeans=KMeans(n_clusters=k, max_iter=1000).fit(y)

    y["clusters"]=kmeans.labels_

    dic[k]=kmeans.inertia_

plt.figure(figsize=(15,10))

plt.plot(list(dic.keys()), list(dic.values()))

plt.show()
# Cluster Customer based on Recency

kmodel_recency=KMeans(n_clusters=4)

kmodel_recency.fit(y)

kpredict_recency=kmodel_recency.predict(y)

kpredict_recency[0:5]

df_recency["RecencyCluster"]=kpredict_recency

df_recency.head()
# get statistical analysis for each cluster

df_recency.groupby(["RecencyCluster"])["Recency"].describe()
# frequency of orders

df_frequency=df.groupby(["CustomerID"])["InvoiceDate"].count().reset_index()

df_frequency.columns=["CustomerID", "Frequency"]

df_frequency=pd.merge(df_user, df_frequency, on="CustomerID")

df_frequency.head()
# Review of Frequency Distribution

plt.figure(figsize=(15,10))

sns.distplot(df_frequency.query("Frequency<1000")["Frequency"])

plt.title("Frequency Distribution")

plt.xlabel("Frequency")

plt.ylabel("Count")
# Customer Segmentation based on Frequency

x=df_frequency[["Frequency"]]

k_model_frequency=KMeans(n_clusters=4)

k_model_frequency.fit(x)

k_model_frequency_predict=k_model_frequency.predict(x)

df_frequency["FrequencyCluster"]=k_model_frequency_predict

df_frequency.head()
# Statistical Analysis of clusters based on frequency

df_frequency.groupby(["FrequencyCluster"])["Frequency"].describe()
df_customer_revenue=df.groupby(["CustomerID"])["Revenue"].sum().reset_index()

df_customer_revenue=pd.merge(df_user, df_customer_revenue, on="CustomerID")

df_customer_revenue.head()
# Revenue Distribution

plt.figure(figsize=(15,10))

sns.distplot(df_customer_revenue.query("Revenue < 10000")["Revenue"])
# Segmenting Customers Based on their Monetary Value

a=df_customer_revenue[["Revenue"]]

k_model_revenue=KMeans(n_clusters=4)

k_model_revenue.fit(a)

k_model_revenue_pred=k_model_revenue.predict(a)

df_customer_revenue["RevenueCluster"]=k_model_revenue_pred

df_customer_revenue.groupby(["RevenueCluster"])["Revenue"].describe()