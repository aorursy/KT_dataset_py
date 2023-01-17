# Import Libraries



import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/Mall_Customers.csv')
df.head()
df['Gender'] = df['Gender'].astype('category')

df['Gender'] = df['Gender'].cat.codes
df.head()
df.isnull().sum()
df.dtypes
fig,ax = plt.subplots(figsize = (8,5))

sns.heatmap(df.corr(), ax = ax, annot = True, linewidths= 0.05, fmt = '.2f',cmap='PuBuGn',linecolor='y')

plt.show()
plt.scatter(df['Gender'],df['Spending Score (1-100)'])

plt.ylabel('spending score')

plt.xlabel('gender')

plt.title('Spending score vs age')

plt.show()
x = df['Age']

y= df['Spending Score (1-100)']

plt.scatter(x,y)

plt.ylabel('spending score')

plt.xlabel('Age')

plt.title('Spending score vs age')

plt.show()
x = df['Annual Income (k$)']

y= df['Spending Score (1-100)']

plt.scatter(x,y)

plt.ylabel('spending score')

plt.xlabel('Annual Income')

plt.title('Spending score vs income')

plt.show()
Customer_id = df['CustomerID']

df.drop(['CustomerID'],axis =1,inplace = True)
x = df
# find n_clusters value using elbow method



from sklearn.cluster import KMeans



a = []

for k in range (1,12):

    kmeans = KMeans(n_clusters = k)

    kmeans.fit(x)

    a.append(kmeans.inertia_)



plt.plot(range(1,12),a)

plt.xlabel('k values')

plt.ylabel('a')

plt.show()
#n_clusters= 5



kmeans = KMeans(n_clusters= 5)

kmeans.fit(x)
clusters_knn = kmeans.fit_predict(x)
plt.scatter(x[clusters_knn == 0]['Annual Income (k$)'],x[clusters_knn == 0]['Spending Score (1-100)'], color = 'Red')

plt.scatter(x[clusters_knn == 1]['Annual Income (k$)'],x[clusters_knn == 1]['Spending Score (1-100)'], color = 'Blue')

plt.scatter(x[clusters_knn == 2]['Annual Income (k$)'],x[clusters_knn == 2]['Spending Score (1-100)'], color = 'Green')

plt.scatter(x[clusters_knn == 3]['Annual Income (k$)'],x[clusters_knn == 3]['Spending Score (1-100)'], color = 'lightcoral')

plt.scatter(x[clusters_knn == 4]['Annual Income (k$)'],x[clusters_knn == 4]['Spending Score (1-100)'], color = 'deepskyblue')



plt.scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:, 3], s = 300, c = 'yellow', label = 'Centroids')

plt.title('Clusters of customers')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')



plt.show()
#"Red" cluster indicates Low Annual income but Spending High.

#"Blue" cluster indicates High Annual income but Spending Low.

#"Green" cluster indicates High Annual income and High Spending.

#"lightcoral" indicates Spending is proportional to Annual income

#"deepskyblue" indicates Low Annual income and Low Spending