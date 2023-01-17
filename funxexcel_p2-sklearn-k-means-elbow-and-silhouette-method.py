import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')

df.head()
df.rename(columns={'Annual Income (k$)' : 'Income', 'Spending Score (1-100)' : 'Spending_Score'}, inplace = True)

df.head()
df_Short = df[['Spending_Score','Income']]
import sklearn.cluster as cluster
K=range(1,12)

wss = []

for k in K:

    kmeans=cluster.KMeans(n_clusters=k,init="k-means++")

    kmeans=kmeans.fit(df_Short)

    wss_iter = kmeans.inertia_

    wss.append(wss_iter)
mycenters = pd.DataFrame({'Clusters' : K, 'WSS' : wss})

mycenters
sns.scatterplot(x = 'Clusters', y = 'WSS', data = mycenters, marker="+")

# We get 5 Clusters
import sklearn.metrics as metrics
for i in range(3,13):

    labels=cluster.KMeans(n_clusters=i,init="k-means++",random_state=200).fit(df_Short).labels_

    print ("Silhouette score for k(clusters) = "+str(i)+" is "

           +str(metrics.silhouette_score(df_Short,labels,metric="euclidean",sample_size=1000,random_state=200)))
# We will use 2 Variables for this example

kmeans = cluster.KMeans(n_clusters=5 ,init="k-means++")

kmeans = kmeans.fit(df[['Spending_Score','Income']])
df['Clusters'] = kmeans.labels_
sns.scatterplot(x="Spending_Score", y="Income",hue = 'Clusters',  data=df)