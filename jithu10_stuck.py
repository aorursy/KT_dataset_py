
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import category_encoders as ce
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

data= pd.read_csv("../input/ctrust/ctrust.csv")
data.head()
df = pd.DataFrame(data)
dff= pd.DataFrame(data)
df['score'].replace({'untrustworthy':0,'trustworthy':1},inplace=True)

dff = pd.DataFrame({
    'context':["sport", "game", "Ecommerece", "holiday", "game", "sport"], 
    'outcome':[1, 2, 3, 4, 2, 1]})

X = dff.drop('outcome', axis = 1)
Y = dff.drop('context', axis = 1)

ce_one_hot= ce.OneHotEncoder(cols= 'context')
ce_one_hot.fit_transform(X,Y)


df['context'].replace({'sport':1,'game':2,'ECommerce':3,'holiday':4},inplace=True)

df.head()
# Basic data Analysis
print(df.info())
print(df.shape)
print(df.describe())
print(df["context"].value_counts())
#plotting 
X=df.iloc[:,3]
Y=df.iloc[:,0]
Z=df.iloc[:,1]
A=df.iloc[:,2]

plt.xlabel('Transaction context') 


#plt.scatter(X,Y)
plt.ylabel('count trust')
plt.scatter(X, Y)

#plt.scatter(X,Z)
plt.ylabel('count untrust')
plt.scatter(X, Z)

#plt.scatter(X,A)
plt.ylabel('last time')
plt.scatter(X, A)

df.head()
# let's check the shape of x
x = df.iloc[:, [2, 3]].values

print(x.shape)

# elbow method
wcss = []
for i in range(1, 10):
    km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    km.fit(x)
    wcss.append(km.inertia_)
    
plt.plot(range(1, 10), wcss)
plt.title('The Elbow Method', fontsize = 20)
plt.xlabel('No. of Clusters')
plt.ylabel('wcss')
plt.show()
#Silhouette Method

no_of_clusters = [2, 3, 4, 5, 6] 
print("Average Silhouette Method\n")
for n_clusters in no_of_clusters: 
  
    cluster = KMeans(n_clusters = n_clusters) 
    cluster_labels = cluster.fit_predict(X) 
  
    # The silhouette_score gives the  
    # average value for all the samples. 
    silhouette_avg = silhouette_score(X, cluster_labels) 
  
    print("For no of clusters =", n_clusters," The average silhouette_score is :", silhouette_avg)

#KMeans Clustering

kmean = KMeans(n_clusters=3, random_state=0).fit(X)
y_kmeans = kmean.predict(X)
lab = kmean.labels_
#for i in range(1,10):
plt.figure(figsize=(14,7))
plt.title("KMeans Clustering on Destinations Rating",fontsize=20)
plt.scatter(X[:,0], X[:,1],c = y_kmeans, s=80, cmap='cividis',alpha=0.8,marker='H')
plt.xlabel("X Axis Rating")
plt.ylabel("Y Axis Rating")
plt.show()
