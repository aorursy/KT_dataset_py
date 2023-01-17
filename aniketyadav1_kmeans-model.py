from sklearn.cluster import KMeans

import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from matplotlib import pyplot as plt

%matplotlib inline
df = pd.read_csv("../input/kmeans/k-means.csv")

df.head()
df.info()
plt.scatter(df.age,df.income)

plt.xlabel('age')

plt.ylabel('income')
from sklearn.cluster import KMeans
km=KMeans(n_clusters=3)

y_predicted=km.fit_predict(df[['age','income']])
y_predicted
df['cluster']=y_predicted

df.head()
km.cluster_centers_
df1=df[df.cluster==0]

df2=df[df.cluster==1]

df3=df[df.cluster==2]
plt.scatter(df1.age,df1['income'],color='red')

plt.scatter(df2.age,df2['income'],color='green')

plt.scatter(df3.age,df3['income'],color='black')

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')

plt.xlabel('age')

plt.ylabel('income')

plt.legend()
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(df[['income']])

df.income=scaler.transform(df[['income']])



scaler.fit(df[['age']])

df.age=scaler.transform(df[['age']])
df.head()
km=KMeans(n_clusters=3)

y_predicted=km.fit_predict(df[['age','income']])
y_predicted
df['cluster']=y_predicted

df.head()
km.cluster_centers_
df1=df[df.cluster==0]

df2=df[df.cluster==1]

df3=df[df.cluster==2]
plt.scatter(df1.age,df1['income'],color='red')

plt.scatter(df2.age,df2['income'],color='green')

plt.scatter(df3.age,df3['income'],color='black')

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')

plt.xlabel('age')

plt.ylabel('income')

plt.legend()
km.inertia_
sse = []

k_rng = range(1,10)

for k in k_rng:

    km = KMeans(n_clusters=k)

    km.fit(df[['age','income']])

    sse.append(km.inertia_)
sse
plt.xlabel('K')

plt.ylabel('Sum of squared error')

plt.plot(k_rng,sse)
from sklearn.datasets import load_iris
iris=load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)

df.head()
df['flower']=iris.target
df.head()
df.drop(['sepal length (cm)','sepal width (cm)','flower'],axis=1,inplace=True)
df.head()
plt.scatter(df[['petal length (cm)']],df[['petal width (cm)']])
scaler.fit(df[['petal length (cm)']])

df['petal length (cm)']=scaler.transform(df[['petal length (cm)']])
scaler.fit(df[['petal width (cm)']])

df['petal width (cm)']=scaler.transform(df[['petal width (cm)']])
df.head()
km=KMeans(n_clusters=3)

y_predicted=km.fit_predict(df[['petal length (cm)','petal width (cm)']])
km.cluster_centers_
y_predicted

df['cluster']=y_predicted
df1=df[df.cluster==0]
df2=df[df.cluster==1]

df3=df[df.cluster==2]
plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color='red')

plt.scatter(df2['petal length (cm)'],df2['petal width (cm)'],color='green')

plt.scatter(df3['petal length (cm)'],df3['petal width (cm)'],color='black')

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1])
sse=[]

k_rng = range(1,10)

for k in k_rng:

  km=KMeans(n_clusters=k)

  km.fit(df[['petal length (cm)','petal width (cm)']])

  sse.append(km.inertia_)

plt.plot(k_rng,sse)

plt.xlabel('k')

plt.ylabel('sum of squared error')