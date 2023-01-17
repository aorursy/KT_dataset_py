# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data =pd.read_csv('../input/heart-disease-uci/heart.csv')
data =data.head(120) #datamın ilk 120 verisini aldım
data.head(20)
data.columns
data2=data.loc[:,['age','trestbps']]#verimi 14 kolondan 2 kolona düsürüyorum

data3=data2
data2
plt.scatter(data2.age,data2.trestbps) 

plt.xlabel('agee')

plt.ylabel('trestbps')

plt.show()
from sklearn.cluster import KMeans 

wcss = []

for i in range(1,15):

    kmeans2 =KMeans(n_clusters=i)

    kmeans2.fit(data2)

    wcss.append(kmeans2.inertia_)



plt.plot(range(1,15),wcss)

plt.xlabel('k_values')

plt.ylabel('k_values_output')

plt.show()  
from sklearn.cluster import KMeans 

kmeans =KMeans(n_clusters = 3)

kmeans.fit(data2)

new_label =kmeans.predict(data2) #3 boyuta göre sınıflandırdım verilerimi =0,1,2 diye

data2['label']=new_label #sınıflandırma sonuclarımı label diye bir kolonlo datama ekliyorum
plt.scatter(data2.age[data2.label ==0],data.trestbps[data2.label ==0],color ='red')

plt.scatter(data2.age[data2.label ==1],data.trestbps[data2.label ==1],color ='green')

plt.scatter(data2.age[data2.label ==2],data.trestbps[data2.label ==2],color ='blue')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color ='yellow')#centers degerlerim

plt.show()
data
from scipy.cluster.hierarchy import linkage,dendrogram

#dendogram çizdirecegim sekilin ismi ,linkage =dendogramı çizdirirken kullanacagım: HC algoritması

merg =linkage(data3,method='ward')#ward classların içindeki yayılımı minimize etmeye yarıyor

dendrogram(merg,leaf_rotation =90)#x eksenindeki yazıları 90'ar derece ile yazdırıyorum

plt.xlabel('data points')

plt.ylabel('data euclidean_distance')

plt.show()
from sklearn.cluster import AgglomerativeClustering#agglo... =birbirne en yakın verileri 2 li birlestirme ile bütüne ulasma yöntemi

hiyerartical_cluster =AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='ward')#euclidean = square(y^2+x^2)=(3,4,5)

cluster =hiyerartical_cluster.fit_predict(data3) #sınıflandırılmıs verimi bir kolona attım 

data3['label']=cluster#sınıflandırılmıs verimi dataframeme ekledim

plt.scatter(data3.age[data3.label==0],data3.trestbps[data3.label ==0],color ='red')

plt.scatter(data3.age[data3.label==1],data3.trestbps[data3.label ==1],color ='green')