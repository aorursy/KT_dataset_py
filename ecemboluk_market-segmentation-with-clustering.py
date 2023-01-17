import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#data visualization

import matplotlib.pyplot as plt

import seaborn as sns 



#clustering model library

from sklearn.cluster import KMeans

from scipy.cluster.hierarchy import linkage, dendrogram

from sklearn.cluster import AgglomerativeClustering



import os

print(os.listdir("../input"))
#read data

data = pd.read_csv('../input/Mall_Customers.csv')
data.head()
data.info()
print(pd.isnull(data).sum())
data.describe()
data.corr()
plt.figure(figsize=(7,7))

sns.heatmap(data.corr(), annot=True)

plt.show()
labels = ['Male','Female']

sizes = [data.query('Gender == "Male"').Gender.count(),data.query('Gender == "Female"').Gender.count()]

#colors

colors = ['#ffdaB9','#66b3ff']

#explsion

explode = (0.05,0.05)

plt.figure(figsize=(8,8)) 

my_circle=plt.Circle( (0,0), 0.7, color='white')

plt.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85,explode=explode)

p=plt.gcf()

plt.axis('equal')

p.gca().add_artist(my_circle)

plt.show()
plt.figure(figsize=(20,10))

sns.countplot(data.Age)

plt.xlabel("Age")

plt.ylabel("Person Count")

plt.show()
plt.figure(figsize=(20,7))

gender = ['Male', 'Female']

for i in gender:

    plt.scatter(x='Age',y='Annual Income (k$)', data=data[data['Gender']==i],s = 200 , alpha = 0.5 , label = i)

plt.legend()

plt.xlabel("Age")

plt.ylabel("Annual Income (k$)")

plt.title("Annual Income according to Age")

plt.show()
plt.figure(figsize=(20,7))

gender = ['Male', 'Female']

for i in gender:

    plt.scatter(x='Age',y='Spending Score (1-100)', data=data[data['Gender']==i],s = 200 , alpha = 0.5 , label = i)

plt.legend()

plt.xlabel("Age")

plt.ylabel("Spending Score (1-100)")

plt.title("Spending Score according to Age")

plt.show()
plt.figure(figsize=(20,7))

gender = ['Male', 'Female']

for i in gender:

    plt.scatter(x='Annual Income (k$)',y='Spending Score (1-100)', data=data[data['Gender']==i],s = 200 , alpha = 0.5 , label = i)

plt.legend()

plt.xlabel("Annual Income (k$)")

plt.ylabel("Spending Score (1-100)")

plt.title("Spending Score according to Annual Income")

plt.show()
#define k value

wcss = []

data_model = data.drop(['Gender','CustomerID'],axis=1)

for k in range(1,15):

    kmeans = KMeans(n_clusters=k)

    kmeans.fit(data_model)

    wcss.append(kmeans.inertia_)



# the best value is elbow value. It's 5.

plt.figure(figsize=(15,5))

plt.plot(range(1,15),wcss)

plt.xlabel("number of k (cluster) value")

plt.ylabel("wcss")

plt.show()
#create model

kmeans = KMeans(n_clusters=5)

data_predict = kmeans.fit_predict(data_model)



plt.figure(figsize=(15,10))

plt.scatter( x = 'Annual Income (k$)' ,y = 'Spending Score (1-100)' , data = data_model , c = data_predict , s = 200 )

plt.xlabel("Annual Income (k$)")

plt.ylabel("Spending Score (1-100)")

plt.show()
#create demogram and find the best clustering value

merg = linkage(data_model,method="ward")

plt.figure(figsize=(25,10))

dendrogram(merg,leaf_rotation = 90)

plt.xlabel("data points")

plt.ylabel("euclidean distance")

plt.show()
#create model

hiyerartical_cluster = AgglomerativeClustering(n_clusters = 5,affinity= "euclidean",linkage = "ward")

data_predict = hiyerartical_cluster.fit_predict(data_model)

plt.figure(figsize=(15,10))

plt.scatter( x = 'Annual Income (k$)' ,y = 'Spending Score (1-100)' , data = data_model , c = data_predict , s = 200 )

plt.show()