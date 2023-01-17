# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import plotly.graph_objects as go

import plotly.express as px

from sklearn.metrics import silhouette_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/german-credit/german_credit_data.csv')

df.shape
df.head()
df.tail()
df = df.drop(columns = 'Unnamed: 0')
df.info()
df.describe()
df.isnull().sum()
df['Cicil'] = df['Credit amount'] / df['Duration']
#create correlation with heatmap

corr = df.corr(method = 'pearson')



#convert correlation to numpy array

mask = np.array(corr)



#to mask the repetitive value for each pair

mask[np.tril_indices_from(mask)] = False

fig, ax = plt.subplots(figsize = (15,12))

fig.set_size_inches(20,20)

sns.heatmap(corr, mask = mask, vmax = 0.9, square = True, annot = True)
sns.pairplot(df)
plt.figure(figsize=(16, 6))

sns.countplot(x='Purpose', data=df, order = df['Purpose'].value_counts().index).set_title('Jumlah dan tujuan mengambil kredit')
purpose = df.groupby(by='Purpose').mean()['Age']

purpose_df = pd.DataFrame({'Purpose' : purpose.index, 'Age' : purpose.values.astype(int)})

fig = px.line(purpose_df, x="Purpose", y="Age", title='Rata-rata umur pada setiap tujuan kredit')

fig.show()
df.groupby(by='Housing').mean()
plt.figure(figsize=(16, 6))

sns.distplot(df['Credit amount'], kde = True, color = 'darkblue', label = 'Credit amount').set_title('Distribution Plot of Credit amount')
df_job = df.where(df['Job']==0).dropna()

df_job.where(df_job['Credit amount'] >= df['Credit amount'].mean()).dropna()
from sklearn.cluster import KMeans

import numpy as np
X = np.asarray(df[["Job", "Credit amount", "Duration"]])

X[:,0] = X[:,0] + 1

# X[:,1] = np.log(X[:,1])

# X[:,2] = np.log(X[:,2])

X = np.log(X)
Sum_of_squared_distances = []

K = range(1,15)

for k in K:

    km = KMeans(n_clusters=k)

    km = km.fit(X)

    Sum_of_squared_distances.append(km.inertia_)



plt.figure(figsize=(15,10))

plt.plot(K, Sum_of_squared_distances, 'bx-')

plt.xlabel('k')

plt.ylabel('Sum_of_squared_distances')

plt.title('Elbow Method For Optimal k')

plt.show()
kmeans = KMeans(n_clusters=3)

kmeans.fit(X)

kmeans.labels_
ss_1 = silhouette_score(X, kmeans.labels_, metric='euclidean')
fig = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')

ax = plt.axes(projection="3d")



ax.scatter3D(X[:,0], X[:,1], X[:,2], c=kmeans.labels_, cmap='rainbow')

ax.scatter3D(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], kmeans.cluster_centers_[:,2], c='black')



xLabel = ax.set_xlabel('Jobs', linespacing=3.2)

yLabel = ax.set_ylabel('Credit amount', linespacing=3.1)

zLabel = ax.set_zlabel('Duration', linespacing=3.4)

print("Grafik klasterisasi Jobs - Credit Amount - Duration")
df['Risk Jobs'] = kmeans.labels_
X = np.asarray(df[["Age", "Credit amount", "Duration"]])

X = np.log(X)
Sum_of_squared_distances = []

K = range(1,15)

for k in K:

    km = KMeans(n_clusters=k)

    km = km.fit(X)

    Sum_of_squared_distances.append(km.inertia_)



plt.figure(figsize=(15,10))

plt.plot(K, Sum_of_squared_distances, 'bx-')

plt.xlabel('k')

plt.ylabel('Sum_of_squared_distances')

plt.title('Elbow Method For Optimal k')

plt.show()
kmeans = KMeans(n_clusters=3)

kmeans.fit(X)

kmeans.labels_
ss_2 = silhouette_score(X, kmeans.labels_, metric='euclidean')
fig = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')

ax = plt.axes(projection="3d")



ax.scatter3D(X[:,0], X[:,1], X[:,2], c=kmeans.labels_, cmap='rainbow')

ax.scatter3D(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], kmeans.cluster_centers_[:,2], c='black')



xLabel = ax.set_xlabel('Age', linespacing=3.2)

yLabel = ax.set_ylabel('Credit amount', linespacing=3.1)

zLabel = ax.set_zlabel('Duration', linespacing=3.4)

print("Grafik Klasterisasi Age - Credit Amount - Duration")
df['Risk Ages'] = kmeans.labels_
X = np.asarray(df[["Credit amount", "Duration"]])

X = np.log(X)
Sum_of_squared_distances = []

K = range(1,15)

for k in K:

    km = KMeans(n_clusters=k)

    km = km.fit(X)

    Sum_of_squared_distances.append(km.inertia_)



plt.figure(figsize=(15,10))

plt.plot(K, Sum_of_squared_distances, 'bx-')

plt.xlabel('k')

plt.ylabel('Sum_of_squared_distances')

plt.title('Elbow Method For Optimal k')

plt.show()
kmeans = KMeans(n_clusters=3)

kmeans.fit(X)

kmeans.labels_
ss_3 = silhouette_score(X, kmeans.labels_, metric='euclidean')
fig = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')



plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='rainbow')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color='black')



plt.xlabel('Duration')

plt.ylabel('Credit amount')



print("Grafik klasterisasi Duration - Credit Amount")
df['Risks'] = kmeans.labels_
X = np.asarray(df[["Cicil", "Age"]])

X = np.log(X)
Sum_of_squared_distances = []

K = range(1,15)

for k in K:

    km = KMeans(n_clusters=k)

    km = km.fit(X)

    Sum_of_squared_distances.append(km.inertia_)



plt.figure(figsize=(15,10))

plt.plot(K, Sum_of_squared_distances, 'bx-')

plt.xlabel('k')

plt.ylabel('Sum_of_squared_distances')

plt.title('Elbow Method For Optimal k')

plt.show()
kmeans = KMeans(n_clusters=3)

kmeans.fit(X)

kmeans.labels_
ss_4 = silhouette_score(X, kmeans.labels_, metric='euclidean')
fig = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')



plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='rainbow')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color='black')



plt.xlabel('Cicil')

plt.ylabel('Age')



print("Grafik klasterisasi Cicil - Age")
df['Risk Cicil'] = kmeans.labels_
df.sort_values(by='Cicil', ascending=False).head()
df_result = df.drop(columns=['Cicil', 'Risk Jobs', 'Risk Ages', 'Risks', 'Risk Cicil'])

df_result.head()
print('silhouette Job - Credit amount - Duration: ', ss_1) 

print('silhouette Age - Credit amount - Duration: ', ss_2) 

print('silhouette Credit amount - Duration: ', ss_3) 

print('silhouette Cicil - Age: ', ss_4) 
X = np.asarray(df[["Credit amount", "Duration"]])

X = np.log(X)
silhouette = []

K = range(3,6)

for k in K:

    km = KMeans(n_clusters=k)

    km.fit(X)

    ss = silhouette_score(X, km.labels_, metric='euclidean')

    silhouette.append(ss)

    

pd.DataFrame({'K' : K, 'Silhouette' : silhouette})
km = KMeans(n_clusters=4)

km.fit(X)
df_result['Risk'] = km.labels_
df_result