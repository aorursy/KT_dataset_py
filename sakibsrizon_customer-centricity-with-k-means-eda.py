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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import warnings 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
df = pd.read_csv('../input/Mall_Customers.csv')
df.head(5)
df.describe().T
plt.figure(figsize=(15,5))
plt.subplot(1,3,1,)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.hist(df.Age,bins=30)

plt.subplot(1,3,2)
plt.title('Annual Income Distribution')
plt.xlabel('Annual Income in k$')
plt.ylabel('Count')
plt.hist(df['Annual Income (k$)'],bins=30)

plt.subplot(1,3,3)
plt.title('Spending Score Distribution')
plt.xlabel('Spending Score')
plt.ylabel('Count')
plt.hist(df['Spending Score (1-100)'],bins=30)

plt.tight_layout()

pass
plt.figure(1, figsize=(15,6))
n=0
for x in ['Age','Annual Income (k$)','Spending Score (1-100)']:
    n+=1
    plt.subplot(1,3,n)
    plt.subplots_adjust(hspace=0.5, wspace = 0.5)
    sns.distplot(df[x],bins = 20)
    plt.title('Distplot of {}'.format(x))
plt.show()
plt.figure(figsize=(15,7))
plt.subplot(1,3,1,)
plt.title('Age Vs Spening Score')
plt.xlabel('Age')
plt.ylabel('Spending Score')
plt.scatter(df['Age'],df['Spending Score (1-100)'],s=100,c='g')

plt.subplot(1,3,2)
plt.title('Annual Income Vs Spending Score')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.scatter(df['Annual Income (k$)'],df['Spending Score (1-100)'],s=100,c='b')

colors = ['g','r','m','b']

plt.subplot(1,3,3,)
plt.title('Gender Vs Spending Score')
plt.xlabel('Gender')
plt.ylabel('Spending Score')
cat_list=df['Gender'].unique()
cat_average=df.groupby('Gender').mean()['Spending Score (1-100)']
plt.bar(cat_list,cat_average,color=colors)

plt.tight_layout()

pass
plt.figure(1,figsize=(15,7))
n=0
for x in ['Age','Annual Income (k$)','Spending Score (1-100)']:
    for y in ['Age','Annual Income (k$)','Spending Score (1-100)']:
        n+=1
        plt.subplot(3,3,n)
        plt.subplots_adjust(hspace = 0.5, wspace= 0.5)
        sns.regplot(x=x,y=y,data = df)
        plt.ylabel(y.split()[0]+''+y.split()[1] if len(y.split())> 1 else y)
plt.show()
X = df.iloc[:,[3,4]].values
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init = 'k-means++', max_iter = 300, n_init = 10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()
kmeans= KMeans(n_clusters=5, init = 'k-means++', max_iter= 300, n_init = 10, random_state=0)
y_kmeans = kmeans.fit_predict(X)
plt.scatter(X[y_kmeans == 0,0],X[y_kmeans==0,1], s= 100, c='magenta', label = 'Careful')
plt.scatter(X[y_kmeans == 1,0],X[y_kmeans==1,1], s= 100, c='yellow', label = 'Standard')
plt.scatter(X[y_kmeans == 2,0],X[y_kmeans==2,1], s= 100, c='green', label = 'Target')
plt.scatter(X[y_kmeans == 3,0],X[y_kmeans==3,1], s= 100, c='cyan', label = 'careless')
plt.scatter(X[y_kmeans == 4,0],X[y_kmeans==4,1], s= 100, c='burlywood', label = 'Sensible')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],s = 300, c='red', label ='Centroids')
plt.title('Cluster of Clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show
pass
X1 = df.iloc[:,[2,4]].values
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init = 'k-means++', max_iter = 300, n_init = 10, random_state=0)
    kmeans.fit(X1)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()
plt.scatter(X1[y_kmeans==0,0],X1[y_kmeans==0,1],s=100,c='magenta',label='Low spenders ')
plt.scatter(X1[y_kmeans==1,0],X1[y_kmeans==1,1],s=100,c='blue',label='Young High Spenders')
plt.scatter(X1[y_kmeans==2,0],X1[y_kmeans==2,1],s=100,c='green',label='Young Average Spenders')
plt.scatter(X1[y_kmeans==3,0],X1[y_kmeans==3,1],s=100,c='cyan',label='Old Average Spenders')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='red',label='Centroids')
plt.title('Cluster of Clients')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.ioff()
plt.show
pass
plt.plot(df['Annual Income (k$)'],df['Age'])
plt.title('Age vs Annual Income')
plt.xlabel('Annual Income')
plt.ylabel('Age')
plt.show()
X = df['Age'].values
y = df['Annual Income (k$)'].values
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y, test_size = 1/2, random_state= 0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
plt.scatter(X_train, y_train, color = 'red')
plt.scatter(X_test, regressor.predict(y_test), color= 'blue')
plt.title('Age Vs Annual Income')
plt.xlabel('Age')
plt.ylabel('Annual Income')
plt.show()
