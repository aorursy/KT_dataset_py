import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.cluster import KMeans
df = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
df.head()
df.describe()
df.drop(columns='CustomerID', axis=1, inplace=True)
df.isna().sum()
df.rename(columns={"Gender":"gender","Age":"age","Annual Income (k$)":"income","Spending Score (1-100)":"score"},inplace=True)
plt.figure(figsize=(7,4))

sns.set(style="whitegrid")

sns.boxplot(x='gender',y='score', data=df)

plt.title("GENDER X SCORE")
plt.figure(figsize=(7,4))

sns.regplot(x='age',y='score', data=df)

plt.title("Age X SCORE")
plt.figure(figsize=(7,4))

sns.regplot(x='income',y='score', data=df)

plt.title("Income X SCORE")
df['gender'].value_counts()
df.corr()
le = LabelEncoder()

le.fit(df['gender'])

df['gender'] = le.transform(df['gender'])
x = df[['age','gender','income','score']]

scaler = StandardScaler().fit(x)

x = scaler.transform(x)
errors = []

for i in range(1,10):

    k = KMeans(n_clusters=i, init='k-means++')

    k.fit(x)

    errors.append(k.inertia_)
plt.plot(range(1,10),errors)

plt.annotate('N',xy=(3.8,350))
kmeans = KMeans(n_clusters=4, init='k-means++')

kmeans.fit(x)
#get the labels and centroids from machine learning algorithm K-MEANS

labels = kmeans.labels_

centroids = kmeans.cluster_centers_
#Apply the labels to the dataframe

df['cluster'] = labels
sns.boxplot(x='cluster',y='score',data=df)
colors = ['slateblue','firebrick','lightsteelblue','tomato']

clusters = df['cluster'].unique()

plt.figure(figsize=(10,8))

for i, col in zip(range(len(centroids)),colors):

    members = df['cluster'] == i

    centroid = centroids[i]

    plt.scatter(df[members]['age'],df[members]['score'],c = col, label='Cluster{}'.format(i))

plt.ylabel('Score', fontsize='large')

plt.xlabel('Age', fontsize='large')

plt.title('Cluster Age X Score', fontsize='x-large', y=1.020)

plt.legend(loc='upper right', fontsize='large')    
colors = ['slateblue','firebrick','lightsteelblue','tomato']

clusters = df['cluster'].unique()

plt.figure(figsize=(10,8))

for i, col in zip(range(len(centroids)),colors):

    members = df['cluster'] == i

    centroid = centroids[i]

    plt.scatter(df[members]['income'],df[members]['score'],c = col, label='Cluster{}'.format(i))

plt.ylabel('Score', fontsize='large')

plt.xlabel('Anual income', fontsize='large')

plt.title('Cluster Income X Score', fontsize='x-large', y=1.020)

plt.legend(loc='upper right', fontsize='large')    
df[df['cluster'] == 1].describe()
df[df['cluster'] == 2].describe()