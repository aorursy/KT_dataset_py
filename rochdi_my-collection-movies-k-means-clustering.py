# libraries

import numpy as np

import pandas as pd

import pylab as pl

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_excel('/kaggle/input/movies_collection.xlsx')
df.head()
# transform categorical data

cols_to_transform = ['Seen']

 

df = pd.get_dummies( columns = cols_to_transform , data = df)

# Seen --- 2 value 0 or 1 

# we gonna remove one column

df = df.drop(columns=['Seen_False'])
df_numeric = df[['Orginal Name','Seen_True',

                 'Budget','Year','Duration',

                 'Votes','Rating' ,

                 'Personal Rating']]
df_numeric.head()
df_numeric.isnull().sum()
df_numeric.dropna(inplace=True)
df_numeric['Budget'].describe()
df_numeric['Duration'].describe()
df_numeric['Votes'].describe()
df_numeric['Rating'].describe()
df_numeric['Personal Rating'].describe()
from sklearn import preprocessing

minmax_processed = preprocessing.MinMaxScaler().fit_transform(df_numeric.drop('Orginal Name',axis=1))

df_numeric_scaled = pd.DataFrame(minmax_processed, index=df_numeric.index, columns=df_numeric.columns[:-1])

df_numeric_scaled.head()
Nc = range(1, 20)

kmeans = [KMeans(n_clusters=i) for i in Nc]
score = [kmeans[i].fit(df_numeric_scaled).score(df_numeric_scaled) for i in range(len(kmeans))]
pl.plot(Nc,score)

pl.xlabel('Number of Clusters')

pl.ylabel('Score')

pl.title('Elbow Curve')

pl.show()
kmeans = KMeans(n_clusters=5)

kmeans.fit(df_numeric_scaled)
len(kmeans.labels_)
df_numeric['cluster'] = kmeans.labels_
df_numeric.head()
plt.figure(figsize=(12,7))

axis = sns.barplot(x=np.arange(0,5,1),y=df_numeric.groupby(['cluster']).count()['Budget'].values)

x=axis.set_xlabel("Cluster Number")

x=axis.set_ylabel("Number of movies")
df_numeric.groupby(['cluster']).mean()
size_array = list(df_numeric.groupby(['cluster']).count()['Budget'].values)
size_array
df_numeric[df_numeric['cluster']==size_array.index(sorted(size_array)[0])].sample(5)
df_numeric[df_numeric['cluster']==size_array.index(sorted(size_array)[1])].sample(5)
df_numeric[df_numeric['cluster']==size_array.index(sorted(size_array)[-1])].sample(5)

y_kmeans = kmeans.fit_predict(df_numeric_scaled)



X = df_numeric_scaled.as_matrix(columns=None)



# Plot the 5 clusters



from sklearn.datasets.samples_generator import make_blobs

X, y_true = make_blobs(n_samples=300, centers=5,

                       cluster_std=0.60, random_state=0)

plt.scatter(X[:, 0], X[:, 1], s=50)
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5)

kmeans.fit(X)

y_kmeans = kmeans.predict(X)



plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')



centers = kmeans.cluster_centers_

plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)