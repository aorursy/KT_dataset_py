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
df.describe()
#Plot Age, Income and Spending Score Correlation

sns.pairplot(df[['Age','Income', 'Spending_Score']])
import sklearn.cluster as cluster
# We will use 2 Variables for this example

kmeans = cluster.KMeans(n_clusters=5 ,init="k-means++")

kmeans = kmeans.fit(df[['Spending_Score','Income']])
kmeans.cluster_centers_
df['Clusters'] = kmeans.labels_
df.head()
df['Clusters'].value_counts()
df.to_csv('mallClusters.csv', index = False)
sns.scatterplot(x="Spending_Score", y="Income",hue = 'Clusters',  data=df)