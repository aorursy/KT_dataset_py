import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
plt.style.available
plt.style.use('seaborn-dark')
data = pd.read_csv('../input/Mall_Customers.csv')
data.head()
data.shape
data.describe()
data['Gender'].value_counts().plot(kind='bar', title='Plotting records by borough', figsize=(10, 4),align='center')
fig, ax = plt.subplots(figsize=(10,5))

sns.distplot(data['Annual Income (k$)'],

             hist=False,

             kde_kws={'shade':True},

            ax = ax)

ax.axvline(x= 66, color='m', linestyle='--', linewidth=2)
fig, ax = plt.subplots(figsize=(7,5))

ax.scatter(data["Annual Income (k$)"], data["Spending Score (1-100)"], c=data.index)

ax.set_xlabel("Annual Income")

ax.set_ylabel('Spedning Score')
gender = pd.get_dummies(data['Gender'])
gender = gender['Male']
data = pd.concat((data,gender),axis=1)
data.drop(['Gender'],axis=1,inplace=True)
data.head()
y = data['Spending Score (1-100)'].values

X = data[['Age','Annual Income (k$)','Male']].values
from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split
kmeans = KMeans(n_clusters=3)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)
kmeans.fit(X_train,y_train)
y_pred = kmeans.predict(X_test)
print(kmeans.labels_)
clusters = kmeans.fit_predict(X)
fig, ax = plt.subplots(figsize=(10,5))

ax.scatter(X[clusters == 0, 0], X[clusters == 0, 1], s = 80, c = 'Red')

ax.scatter(X[clusters == 1, 0], X[clusters == 1, 1], s = 80, c = 'Blue')

ax.scatter(X[clusters == 2, 0], X[clusters == 2, 1], s = 80, c = 'Green')

ax.legend(['Cluster 1','Cluster 2','Cluster 3'])