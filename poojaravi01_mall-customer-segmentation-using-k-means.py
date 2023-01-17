from sklearn.cluster import KMeans

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import silhouette_score
df=pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')

df.head()
df.shape
df.describe()
df.rename(columns={'Spending Score (1-100)':'SpendingScore','Annual Income (k$)':'AnnualIncome'},inplace=True)

df.head()
sns.countplot(data=df,x='Gender',palette='Set2')
plt.figure(figsize=(15,10))

plt.subplot(3,3,1)

sns.distplot(df['Age'])

plt.subplot(3,3,2)

sns.distplot(df['AnnualIncome'],color='red')

plt.subplot(3,3,3)

sns.distplot(df['SpendingScore'],color='green')
sns.heatmap(df.iloc[:,1:5].corr(),annot=True,linewidths=0.2)
plt.figure(figsize=(10,7))

sns.lineplot(x='AnnualIncome',y='SpendingScore',hue='Gender',data=df,ci=False,style='Gender',markers=True)
plt.figure(figsize=(10,7))

sns.lineplot(x='Age',y='SpendingScore',hue='Gender',data=df,ci=False,style='Gender',markers=True)
plt.figure(figsize=(20,10))

x=0

for i in ['AnnualIncome','SpendingScore']:

    x=x+1

    plt.subplot(2,2,x)

    sns.boxplot(data=df,x=i,y='Gender',palette='Set'+str(x))

plt.show()
lenc=LabelEncoder()

df['Gender']=lenc.fit_transform(df['Gender'])
df.isna().sum()
df.drop('CustomerID',axis=1,inplace=True)
df.head()
cluster=list()

for i in range(1,11):

    kmns=KMeans(n_clusters=i)

    kmns.fit(df)

    cluster.append(kmns.inertia_)

plt.figure(figsize=(10,7))

sns.lineplot(x=list(range(1,11)),y=cluster)
n=3

kmeans3=KMeans(n_clusters=n,n_init=10,max_iter=500)

kmeans3.fit(df)
df['clusters']=kmeans3.labels_

kmeans3.cluster_centers_
df.head()
print(silhouette_score(df.iloc[:,0:4],kmeans3.labels_))
plt.figure(figsize=(12, 8))

sns.scatterplot(df['AnnualIncome'], df['SpendingScore'], hue=df['clusters'], palette='Set1',style=df['Gender'])
n=5

kmeans5=KMeans(n_clusters=n,n_init=10,max_iter=500)

kmeans5.fit(df)
df['clusters']=kmeans5.labels_

kmeans5.cluster_centers_
df.head()
print(silhouette_score(df.iloc[:,0:4],kmeans5.labels_))
plt.figure(figsize=(12, 8))

sns.scatterplot(df['AnnualIncome'], df['SpendingScore'], hue=df['clusters'], palette='Set1',style=df['Gender'])