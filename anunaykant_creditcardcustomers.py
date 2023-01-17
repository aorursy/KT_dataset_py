
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os

"""
Following is the Data Dictionary for Credit Card dataset :-

CUSTID : Identification of Credit Card holder (Categorical)
BALANCE : Balance amount left in their account to make purchases 
BALANCEFREQUENCY : How frequently the Balance is updated, score between 0 and 1 (1 = frequently updated, 0 = not frequently updated)
PURCHASES : Amount of purchases made from account
ONEOFFPURCHASES : Maximum purchase amount done in one-go
INSTALLMENTSPURCHASES : Amount of purchase done in installment
CASHADVANCE : Cash in advance given by the user
PURCHASESFREQUENCY : How frequently the Purchases are being made, score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased)
ONEOFFPURCHASESFREQUENCY : How frequently Purchases are happening in one-go (1 = frequently purchased, 0 = not frequently purchased)
PURCHASESINSTALLMENTSFREQUENCY : How frequently purchases in installments are being done (1 = frequently done, 0 = not frequently done)
CASHADVANCEFREQUENCY : How frequently the cash in advance being paid
CASHADVANCETRX : Number of Transactions made with "Cash in Advanced"
PURCHASESTRX : Numbe of purchase transactions made
CREDITLIMIT : Limit of Credit Card for user
PAYMENTS : Amount of Payment done by user
MINIMUM_PAYMENTS : Minimum amount of payments made by user
PRCFULLPAYMENT : Percent of full payment paid by user
TENURE : Tenure of credit card service for user

We are going to utilise clustering techniques to segment the customers based on their credit card details
"""


df = pd.read_csv("../input/ccdata/CC GENERAL.csv")
df.head(10)
df.shape

df.describe()
df.info()
df = df.drop({'CUST_ID'},axis=1)
df.head(10)
datatypes = {}
for column in df.columns:
    datatypes[column] = float
    
df = df.astype(datatypes)    
df.info()    
df = df.dropna()
df.info()
df = df.drop({'BALANCE_FREQUENCY','TENURE'},axis=1)
df.head(2)
df.mean(axis=0) 
#We will drop the columns with very low mean values here as their significance is very less.
df = df.drop({"PURCHASES_FREQUENCY","ONEOFF_PURCHASES_FREQUENCY","PURCHASES_INSTALLMENTS_FREQUENCY","CASH_ADVANCE_FREQUENCY","PRC_FULL_PAYMENT",    "CASH_ADVANCE_TRX","PURCHASES_TRX"},axis=1)
df.head()
scaler = StandardScaler()
X = scaler.fit_transform(df)
X.shape
n_clusters = 25
inertia = []
for n in range(1,n_clusters):
    km = KMeans(n)
    km.fit(X)
    inertia.append(km.inertia_)
        
plt.plot(inertia)    
km = KMeans(10)
km.fit(X)
km.inertia_
df_cluster = pd.concat([df, pd.DataFrame({'Cluster': km.labels_})],axis=1)
df_cluster = df_cluster.dropna()
df_cluster.info()
df_cluster.head()


arr_0 = df_cluster[df_cluster["Cluster"] == 0.0]
arr_1 = df_cluster[df_cluster["Cluster"] == 1.0]
arr_2 = df_cluster[df_cluster["Cluster"] == 2.0]
arr_3 = df_cluster[df_cluster["Cluster"] == 3.0]
arr_4 = df_cluster[df_cluster["Cluster"] == 4.0]
arr_5 = df_cluster[df_cluster["Cluster"] == 5.0]
arr_6 = df_cluster[df_cluster["Cluster"] == 6.0]
arr_7 = df_cluster[df_cluster["Cluster"] == 7.0]
arr_8 = df_cluster[df_cluster["Cluster"] == 8.0]
arr_9 = df_cluster[df_cluster["Cluster"] == 9.0]
plt.figure(figsize=(10,15))
for c in df:
    arr = []
    arr.append(arr_0[c].mean())
    arr.append(arr_1[c].mean())
    arr.append(arr_2[c].mean())
    arr.append(arr_3[c].mean())
    arr.append(arr_4[c].mean())
    arr.append(arr_5[c].mean())
    arr.append(arr_6[c].mean())
    arr.append(arr_7[c].mean())
    arr.append(arr_8[c].mean())
    arr.append(arr_9[c].mean())
    
    plt.plot(arr,label = c)
    plt.xlabel("Clusters")
    plt.ylabel("Mean_values")
    plt.legend(loc=2,prop = {'size':8})
    
    
    
    
    
    
    
"""
Trends: Across all clusters.
We can see that credit_card limit(brown) and balances(blue) are directly linked.
The purchases(orange) and cash_advance(violet) are directly linked.
No clear trends between payments(pink) and purchases(orange).
Installments purchases(red) are not linked with balance or purchases,but are inversely linked with oneoff purchases(green).
"""
scaler = StandardScaler()
X_PCA = scaler.fit_transform(df)
X_PCA

n_dim = 7
explained_variance  = []
for n in range(1,n_dim):
    pca = PCA(n)
    pca.fit_transform(X_PCA)
    explained_variance.append(pca.explained_variance_ratio_.sum())
    
plt.plot(explained_variance)    
pca = PCA(4)
X_PCA = pca.fit_transform(X_PCA)
pca.explained_variance_ratio_

pca.explained_variance_ratio_.sum()
n_clusters = 25
inertia = []
for n in range(1,n_clusters):
    km = KMeans(n)
    km.fit_transform(X_PCA)
    inertia.append(km.inertia_)
    
plt.plot(inertia)
km = KMeans(10)
km.fit_transform(X_PCA)
km.inertia_
df_cluster_1 = pd.concat([df,pd.DataFrame({"Cluster_PCA": km.labels_})],axis=1)
df_cluster_1 = df_cluster_1.dropna()
df_cluster_1.head()

arr_0 = df_cluster_1[df_cluster_1["Cluster_PCA"] == 0.0]
arr_1 = df_cluster_1[df_cluster_1["Cluster_PCA"] == 1.0]
arr_2 = df_cluster_1[df_cluster_1["Cluster_PCA"] == 2.0]
arr_3 = df_cluster_1[df_cluster_1["Cluster_PCA"] == 3.0]
arr_4 = df_cluster_1[df_cluster_1["Cluster_PCA"] == 4.0]
arr_5 = df_cluster_1[df_cluster_1["Cluster_PCA"] == 5.0]
arr_6 = df_cluster_1[df_cluster_1["Cluster_PCA"] == 6.0]
arr_7 = df_cluster_1[df_cluster_1["Cluster_PCA"] == 7.0]
arr_8 = df_cluster_1[df_cluster_1["Cluster_PCA"] == 8.0]
arr_9 = df_cluster_1[df_cluster_1["Cluster_PCA"] == 9.0]
plt.figure(figsize=(10,15))
for c in df:
    arr = []
    arr.append(arr_0[c].mean())
    arr.append(arr_1[c].mean())
    arr.append(arr_2[c].mean())
    arr.append(arr_3[c].mean())
    arr.append(arr_4[c].mean())
    arr.append(arr_5[c].mean())
    arr.append(arr_6[c].mean())
    arr.append(arr_7[c].mean())
    arr.append(arr_8[c].mean())
    arr.append(arr_9[c].mean())
    
    plt.plot(arr,label = c)
    plt.xlabel("Clusters")
    plt.ylabel("Mean_values")
    plt.legend(loc=2,prop = {'size':8})
    
    
    
"""
We see the same trends here as above.
Only the correlations seem much sharper.
Also here purchases(pink) is much more correlated with balance(blue) and purchases(orange) here.
"""