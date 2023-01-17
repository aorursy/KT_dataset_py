#import libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from yellowbrick.cluster import KElbowVisualizer

import warnings

warnings.filterwarnings("ignore")

import plotly.express as px
#fetch data

df = pd.read_csv('/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')

df.head()
df.info()
#check for null values

df.isnull().sum()
df.describe()
plt.figure(figsize = (12,6))

sns.heatmap(df.drop('CustomerID',axis=1).corr(),annot=True,cmap='viridis',fmt='.2f')
sns.pairplot(df.drop('CustomerID',axis=1),hue='Gender')
plt.figure(figsize=(10,10))

sns.distplot(df['Age'],bins=10,hist_kws=dict(edgecolor="blue"))

plt.show()
plt.figure(figsize=(10,10))

sns.distplot(df['Annual Income (k$)'],color="green",bins=10,hist_kws=dict(edgecolor="green"))

plt.show()
plt.figure(figsize=(10,10))

sns.distplot(df['Spending Score (1-100)'],color="red",bins=10,hist_kws=dict(edgecolor="red"))

plt.show()
plt.figure(figsize=(10,10))

sns.countplot(data=df,x='Gender')

plt.show()
plt.figure(figsize=(10,10))

sns.boxplot(data=df,x='Gender',y='Age')

plt.show()
plt.figure(figsize=(10,10))

sns.boxplot(data=df,x='Gender',y='Annual Income (k$)')

plt.show()
plt.figure(figsize=(10,10))

sns.boxplot(data=df,x='Gender',y='Spending Score (1-100)')

plt.show()
X1 = df[['Age' , 'Spending Score (1-100)']].iloc[: , :]

model = KMeans()

visualizer = KElbowVisualizer(model, k=(1,20))



visualizer.fit(X1)        

visualizer.show()
kmeans1 = KMeans(n_clusters=4)

kmeans1.fit(X1)
kmeans1.cluster_centers_
kmeans1.labels_
sns.set_style(style='darkgrid')

plt.figure(figsize=(14,8))

plt.title('KMeans - Age vs Spending Score (1-100)')

plt.scatter(data=X1,x='Age',y='Spending Score (1-100)',c=kmeans1.labels_,cmap='gist_rainbow')

plt.scatter(kmeans1.cluster_centers_[:, 0], kmeans1.cluster_centers_[:, 1], s=100, c='Black')

plt.show()
X2 = df[['Age' , 'Annual Income (k$)']].iloc[: , :]

model2 = KMeans()

visualizer = KElbowVisualizer(model2, k=(1,20))



visualizer.fit(X2)        

visualizer.show()
kmeans2 = KMeans(n_clusters=4)

kmeans2.fit(X2)
plt.figure(figsize=(14,8))

plt.title('KMeans - Age vs Annual Income (k$)')

plt.scatter(data=X2,x='Age',y='Annual Income (k$)',c=kmeans2.labels_,cmap='gist_rainbow')

plt.scatter(kmeans2.cluster_centers_[:, 0], kmeans2.cluster_centers_[:, 1], s=100, c='Black')

plt.show()
X3 = df[['Annual Income (k$)','Spending Score (1-100)']].iloc[: , :]

model3 = KMeans()

visualizer = KElbowVisualizer(model3, k=(1,20))



visualizer.fit(X3)        

visualizer.show()
kmeans3 = KMeans(n_clusters=5)

kmeans3.fit(X3)
plt.figure(figsize=(14,8))

plt.title('KMeans - Annual Income (k$) vs Spending Score (1-100)')

plt.scatter(data=X3,x='Annual Income (k$)',y='Spending Score (1-100)',c=kmeans3.labels_,cmap='gist_rainbow')

plt.scatter(kmeans3.cluster_centers_[:, 0], kmeans3.cluster_centers_[:, 1], s=100, c='Black')

plt.show()
X4 = df[['Age','Annual Income (k$)','Spending Score (1-100)']].iloc[: , :]

model4 = KMeans()

visualizer = KElbowVisualizer(model4, k=(1,20))



visualizer.fit(X4)        

visualizer.show()
kmeans4 = KMeans(n_clusters=6)

kmeans4.fit(X4)
#Non-Interactive plot

plt.figure(figsize=(14,8))

plt.title('KMeans - Age vs Annual Income (k$) vs Spending Score (1-100)')

ax = plt.axes(projection = '3d')

ax.scatter(X4['Age'],X4['Annual Income (k$)'],X4['Spending Score (1-100)'], c=kmeans4.labels_ , cmap='gist_rainbow', s=100)

ax.scatter(kmeans4.cluster_centers_[:, 0], kmeans4.cluster_centers_[:, 1],kmeans4.cluster_centers_[:, 2], s=100, c='Black')

ax.set_xlabel('Age')

ax.set_ylabel('Annual Income (k$)')

ax.set_zlabel('Spending Score (1-100)')

plt.show()
#Interactive plot

fig = px.scatter_3d(X4, x='Age', y='Annual Income (k$)', z='Spending Score (1-100)',color=kmeans4.labels_,opacity=0.7,)

fig.update(layout_coloraxis_showscale=False)