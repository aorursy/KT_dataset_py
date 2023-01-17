import numpy as np

import pandas as pd
df=pd.read_csv('../input/Mall_Customers.csv')
df.head(5)
missing_data=df.isnull()
missing_data.head(5)
df.shape
for column in missing_data.columns.values.tolist():

    print(column)

    print(missing_data[column].value_counts())

    print('')
df.head(5)
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df.dtypes
df.corr()
df[['Age', 'Annual Income (k$)']].corr()
sns.regplot(x='Age', y='Annual Income (k$)', data=df)

plt.ylim(0,)
df[['Age', 'Spending Score (1-100)']].corr()
sns.regplot(x='Age', y='Spending Score (1-100)', data=df)

plt.ylim(0,)
df[['Spending Score (1-100)','Annual Income (k$)']].corr()
sns.regplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=df)

plt.ylim(0,)
sns.boxplot(x='Genre', y='Annual Income (k$)', data=df)
sns.boxplot(x='Genre', y='Spending Score (1-100)', data=df)
plt.figure(1, figsize=(15,5))

sns.countplot(y='Genre', data=df)

plt.show()
from sklearn.cluster import KMeans

from sklearn.cluster import KMeans

from sklearn.datasets.samples_generator import make_blobs
X1 = df[['Age' , 'Spending Score (1-100)']].iloc[: , :].values

inertia = []

for n in range(1 , 11):

    algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 

                        tol=0.0001,  random_state= 111  , algorithm='elkan') )

    algorithm.fit(X1)

    inertia.append(algorithm.inertia_)
plt.figure(1 , figsize = (15 ,6))

plt.plot(np.arange(1 , 11) , inertia , 'o')

plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)

plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')

plt.show()
algorithm = (KMeans(n_clusters = 4 ,init='k-means++', n_init = 10 ,max_iter=300, 

                        tol=0.0001,  random_state= 111  , algorithm='elkan') )

algorithm.fit(X1)

labels1 = algorithm.labels_

centroids1 = algorithm.cluster_centers_
h = 0.02

x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1

y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) 
plt.figure(1 , figsize = (10 , 5) )

plt.clf()

Z = Z.reshape(xx.shape)

plt.imshow(Z , interpolation='nearest', 

           extent=(xx.min(), xx.max(), yy.min(), yy.max()),

           cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')



plt.scatter( x = 'Age' ,y = 'Spending Score (1-100)' , data = df , c = labels1 , 

            s = 200 )

plt.scatter(x = centroids1[: , 0] , y =  centroids1[: , 1] , s = 300 , c = 'red' , alpha = 0.5)

plt.ylabel('Spending Score (1-100)') , plt.xlabel('Age')

plt.show()