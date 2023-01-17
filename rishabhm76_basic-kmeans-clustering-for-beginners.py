import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.cluster import KMeans

from sklearn import preprocessing



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
raw_data = pd.read_csv('/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')

raw_data.head()
data = raw_data.set_index(['CustomerID'])
mapping = { "Gender" : {"Male":0, "Female":1}} 

data.replace(mapping, inplace=True)
data.head()
data.describe()
data.nunique()
data.isnull().sum()
sns.pairplot(data)
# pair plot with Gender



sns.pairplot(data, hue = 'Gender', kind='reg')
data.corr()
plt.matshow(data.corr())

plt.colorbar()
corr = data.corr()

sns.heatmap(corr, cmap='YlGnBu', linewidths='0.1')
sns.clustermap(corr)
corr.style.background_gradient(cmap='coolwarm')
## Building The Elbow Method Graph with full data



wcss = []

for i in range(1,len(data)):

    kmeans = KMeans(i)

    kmeans.fit(data)

    wcss.append(kmeans.inertia_)

wcss
plt.plot(range(1,len(data)), wcss)
wcss = []

for i in range(1,20):

    kmeans = KMeans(i)

    kmeans.fit(data)

    wcss.append(kmeans.inertia_)

wcss
plt.plot(range(1,20), wcss)
kmeans = KMeans(4)

kmeans.fit(data)

clusters = data.copy()

clusters['prediction'] = kmeans.fit_predict(data)

clusters[:15]
plt.scatter(data['Spending Score (1-100)'],data['Annual Income (k$)'], c= clusters['prediction'], cmap='rainbow')

plt.ylabel('Annual Income (k$)')

plt.xlabel('Spending Score (1-100)')
kmeans = KMeans(5)

kmeans.fit(data)

clusters = data.copy()

clusters['prediction'] = kmeans.fit_predict(data)

plt.scatter(data['Spending Score (1-100)'],data['Annual Income (k$)'], c= clusters['prediction'], cmap='rainbow')

plt.ylabel('Annual Income (k$)')

plt.xlabel('Spending Score (1-100)')
kmeans = KMeans(6)

kmeans.fit(data)

clusters = data.copy()

clusters['prediction'] = kmeans.fit_predict(data)

plt.scatter(data['Spending Score (1-100)'],data['Annual Income (k$)'], c= clusters['prediction'], cmap='rainbow')

plt.ylabel('Annual Income (k$)')

plt.xlabel('Spending Score (1-100)')