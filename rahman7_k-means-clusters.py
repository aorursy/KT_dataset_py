# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# import libraries:
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
# import dataset:
df=pd.read_csv("../input/Mall_Customers.csv")
df.head()
# visualization:
sns.pairplot(df,hue='Genre')
sns.scatterplot(df['Annual Income (k$)'],df['Spending Score (1-100)'],hue='Age',data=df)
sns.boxplot(df['Spending Score (1-100)'],hue='Age',data=df)
sns.boxplot(df['Annual Income (k$)'],hue='Age',data=df)
sns.lineplot(df['Annual Income (k$)'],df['Spending Score (1-100)'],hue='Age',data=df)
sns.lmplot('Annual Income (k$)','Spending Score (1-100)',hue='Age',data=df)
df.hist(column='Annual Income (k$)',figsize=(12,6))
df.hist(column='Spending Score (1-100)',figsize=(10,6))
sns.catplot(x='Annual Income (k$)',y='Spending Score (1-100)',hue='Genre',data=df)
# spliting the dataset into the dependent and endependent:
X=df.iloc[:,[3,4]].values
X
# Useing the Elbow method to find optimal on of the cluster:
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',n_init=10,max_iter=300,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow method of optimal')
plt.xlabel('No of clusters')
plt.ylabel('WCSS')
plt.show()

#Apply the K-means on the Mall dataset:
kmeans=KMeans(n_clusters=5,init='k-means++',n_init=10,max_iter=300,random_state=0)
y_pred=kmeans.fit_predict(X)

# Visualiztion of cluster:
plt.scatter(X[y_pred==0,0],X[y_pred==0,1],s=100,color='red',label='cluster 1')
plt.scatter(X[y_pred==1,0],X[y_pred==1,1],s=100,color='blue',label='cluster 2')
plt.scatter(X[y_pred==2,0],X[y_pred==2,1],s=100,color='yellow',label='cluster 3')
plt.scatter(X[y_pred==3,0],X[y_pred==3,1],s=100,color='cyan',label='cluster 4')
plt.scatter(X[y_pred==4,0],X[y_pred==4,1],s=100,color='magenta',label='cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,color='yellow',label='centroied')
plt.legend()
plt.title('Luster of clint')
plt.xlabel('Anual Income')
plt.ylabel('Salary')
plt.show()

