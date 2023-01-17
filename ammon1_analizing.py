import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
df=pd.read_csv("../input/Mall_Customers.csv")
print(df.head())
df.isnull().values.any()
print('clean dataset')

# Any results you write to the current directory are saved as output.
df=df.drop(['CustomerID'],axis=1)
print(df.columns.values)
df=pd.get_dummies(df, prefix='GENDER_', columns=['Gender'])
print(df.columns.values)
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
X=df.iloc[:,:].values
wcss=[]
for i in range(1,14):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,14),wcss)
kmeans = KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10, random_state=0)
y_kmeans=kmeans.fit_predict(X)
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100, c='red',label='c0')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100, c='green',label='c1')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100, c='purple',label='c2')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100, c='blue',label='c3')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100, c='pink',label='c4')
plt.xlabel('age')
plt.ylabel('income')
plt.show()
print(df.columns.values)
print('Peaple are grouped partially on income'
     'c4 and c0 gropus have same, smallest income'
     'c2 has medium income'
     'c3 and c1 have highest income')
plt.scatter(X[y_kmeans==0,1],X[y_kmeans==0,2],s=100, c='red',label='c0')
plt.scatter(X[y_kmeans==1,1],X[y_kmeans==1,2],s=100, c='green',label='c1')
plt.scatter(X[y_kmeans==2,1],X[y_kmeans==2,2],s=100, c='purple',label='c2')
plt.scatter(X[y_kmeans==3,1],X[y_kmeans==3,2],s=100, c='blue',label='c3')
plt.scatter(X[y_kmeans==4,1],X[y_kmeans==4,2],s=100, c='pink',label='c4')
plt.xlabel('Income')
plt.ylabel('Score')
plt.show()
print("Propably best parameters to sort groups")
plt.scatter(X[y_kmeans==0,2],X[y_kmeans==0,3],s=100, c='red',label='c0')
plt.scatter(X[y_kmeans==1,2],X[y_kmeans==1,3],s=100, c='green',label='c1')
plt.scatter(X[y_kmeans==2,2],X[y_kmeans==2,3],s=100, c='purple',label='c2')
plt.scatter(X[y_kmeans==3,2],X[y_kmeans==3,3],s=100, c='blue',label='c3')
plt.scatter(X[y_kmeans==4,2],X[y_kmeans==4,3],s=100, c='pink',label='c4')
plt.xlabel('Score')
plt.ylabel('Male/Female')
plt.show()
print('It seems male have averagly slightly higher score in all groups')