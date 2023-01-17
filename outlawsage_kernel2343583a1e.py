import pandas as pd

credit_df=pd.read_csv("../input/ccdata/CC GENERAL.csv")

credit_df.columns
from sklearn.preprocessing import StandardScaler



credit_df=credit_df.drop("CUST_ID",axis=1)

credit_df.fillna(method="ffill",inplace=True)



scaler= StandardScaler()

scaled_beer_df=scaler.fit_transform(credit_df[[ 'BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES',

       'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',

       'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',

       'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY',

       'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS',

       'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE']])

credit_df.head(6)
#we'll use elbow curve method to find the optimal no. of clusters

from sklearn.cluster import KMeans

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sn

%matplotlib inline



cluster_range=range(1,10)

cluster_errors=[]



for num_clusters in cluster_range:

    clusters=KMeans(num_clusters)

    clusters.fit(scaled_beer_df)

    cluster_errors.append(clusters.inertia_)

    

plt.figure(figsize=(6,4))

plt.plot(cluster_range,cluster_errors,marker="o");
k=4



clusters=KMeans(k,random_state=42)

clusters.fit(scaled_beer_df)

credit_df["clusterid"]=clusters.labels_
for c in credit_df:

    grid= sn.FacetGrid(credit_df, col='clusterid')

    grid.map(plt.hist, c)
credit_df.groupby('clusterid').mean()