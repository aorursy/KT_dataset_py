import pandas as pd

import numpy as np

import seaborn as sns
df = pd.read_csv("../input/customer-segmentation/customer_segmentation.csv",  sep = ",")
df.head()
df.shape
df.info()
df.isnull().sum()
df.describe().T
df.corr()
def IsiHaritasiCiz():

    corr = df.corr()

    return sns.heatmap(corr, 

                xticklabels=corr.columns.values,

                yticklabels=corr.columns.values);

IsiHaritasiCiz()
sns.countplot(data = df, x = "cinsiyet")
df["cinsiyet"].value_counts().plot.barh()
sns.distplot(df["maas"], bins = 16, color = "red");
sns.distplot(df["aylik_harcama"], bins = 16, color = "red");
sns.scatterplot(x = "yas", y = "maas", data = df, color = "purple");
sns.scatterplot(x = "yas", y = "aylik_harcama", data = df, color = "blue");
sns.scatterplot(x = "maas", y = "aylik_harcama", data = df, color = "blue");
sns.scatterplot(x = "maas", y = "aylik_harcama", data = df, hue="cinsiyet");
IsiHaritasiCiz()
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
sonuclar = []

for i in range(1,11):

    kmeans = KMeans (n_clusters = i, init='k-means++', random_state= 123)

    kmeans.fit(df.values)

    sonuclar.append(kmeans.inertia_)#WCSS DEGERLERİNİ ALIYORUZ.

plt.plot(range(1,11),sonuclar)
kmeans = KMeans(n_clusters = 3, init = 'k-means++')

kmeans.fit(df.values)
merkezler = kmeans.cluster_centers_.astype(int)

merkezler 
kumeler = kmeans.labels_

kumeler = kumeler.tolist()

df["Kümeler"] = kumeler

df.head()
#!pip install plotly

import plotly.express as px

import plotly.graph_objs as go
df["Kümeler"]=df["Kümeler"].astype(str)

fig=px.scatter(df,

               x="maas"

              ,y="aylik_harcama"

              ,color="Kümeler"

              ,title="3'lü kümeleme"

              )

fig.add_trace(go.Scatter(x=merkezler[:,0],y=merkezler[:,1],mode="markers",name="Merkezler",marker=dict(color="black", size = 50)

                        )

             )

fig.show()
from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')

#linkage ile kumeler arası mesafe, affinity ile veriler arası mesafe olcum turu belirleniyor.

X = df.values

Y_tahmin = ac.fit_predict(X)

print(Y_tahmin)



plt.scatter(X[Y_tahmin==0,0],X[Y_tahmin==0,1],s=100, c='red')

plt.scatter(X[Y_tahmin==1,0],X[Y_tahmin==1,1],s=100, c='blue')

plt.scatter(X[Y_tahmin==2,0],X[Y_tahmin==2,1],s=100, c='green')

plt.scatter(X[Y_tahmin==3,0],X[Y_tahmin==3,1],s=100, c='yellow')

plt.title('HC')

plt.show()



import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))

plt.show()
ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')

#linkage ile kumeler arası mesafe, affinity ile veriler arası mesafe olcum turu belirleniyor.

Y_tahmin = ac.fit_predict(X)

print(Y_tahmin)



plt.scatter(X[Y_tahmin==0,0],X[Y_tahmin==0,1],s=100, c='red')

plt.scatter(X[Y_tahmin==1,0],X[Y_tahmin==1,1],s=100, c='blue')



plt.title('HC')

plt.show()
