# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/iris/Iris.csv")
data.head()
data.info()
x=data.iloc[:,1:5].values # İşlem görecek değerler.

y=data.iloc[:,5:].values # Tahmin edilecek değerler.

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)

#Verileri parçaladık.
from sklearn.preprocessing import StandardScaler

ss=StandardScaler()

X_train=ss.fit_transform(x_train)

X_test=ss.transform(x_test)

#X değerlerimizi normalize ettik. 
from sklearn.linear_model import LogisticRegression

logr=LogisticRegression(random_state=0)

logr.fit(X_train,y_train) # Train değerlerine öğrenmesini istedik

y_pred=logr.predict(X_test) # X_test değerine bakıp y_test değerlerine tahmin ettirdik.
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred) # Ne kadar değeri doğru tahmin ettik diye gözlem yapmak için kullandık. (1,1)(2,2)(3,3) doğru değerlerdir.

cm

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier()

knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)



cm=confusion_matrix(y_test,y_pred)

cm
from sklearn.svm import SVC

svc=SVC(kernel="linear")

svc.fit(X_train,y_train)

y_pred=svc.predict(X_test)



cm=confusion_matrix(y_test,y_pred)

cm
from sklearn.naive_bayes import GaussianNB

gnb=GaussianNB()

gnb.fit(X_train,y_train)

y_pred=gnb.predict(X_test)



cm=confusion_matrix(y_test,y_pred)

cm
from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier(criterion="gini")

dtc.fit(X_train,y_train)

y_pred=dtc.predict(X_test)



cm=confusion_matrix(y_test,y_pred)

cm
x
#K-Means Algoritması

from sklearn.cluster import KMeans

km=KMeans(n_clusters=3,init="k-means++")# 3 değerini kafandan vermemiz yanlış sonuca götürebilir. 

km.fit(x) # ilk başta belli değerler ile kontrol etmemiz lazım en iyi çalıştığı değeri öğrenmek için.

sonuclar=[]

for i in range(1,11):

    km=KMeans(n_clusters=i,init="k-means++",random_state=123)# random_state =111 dememizin sebebi aynı değerlerden başlıyor olması. Farklı değerde verilebilir.

    km.fit(x)

    sonuclar.append(km.inertia_)

# Şimdi en iyi seçeneği görmek için sonucları çizdirmemiz lazım

import matplotlib.pyplot as plt

plt.plot(range(1,11),sonuclar)

plt.show()
# Grafiğe bakıldığında en iyi noktamız 3 ile 4 olur ama 4'ten sonra azalma az olduğu için 4 en iyi nokta olacaktır.

# Bu yüzden 4 noktasını seçip tekrar görelim

km=KMeans(n_clusters=4,init="k-means++",random_state=123)

tahmin=km.fit_predict(x)

fig = data[data.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='orange', label='Setosa')

data[data.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue', label='versicolor',ax=fig)

data[data.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=fig)

fig.set_title("İlk baştaki veri dağılımı")

plt.show()
plt.scatter(x[tahmin==0,0],x[tahmin==0,1],color="r",s=100)

plt.scatter(x[tahmin==1,0],x[tahmin==1,1],color="g",s=100)

plt.scatter(x[tahmin==2,0],x[tahmin==2,1],color="b",s=100)

plt.scatter(x[tahmin==3,0],x[tahmin==3,1],color="y",s=100)

plt.title("K-Means uygulandıktan sonraki veri dağılımı")

plt.show()

#Hierarchical Clustering ( Hiyerarşik Kümeleme )

from sklearn.cluster import AgglomerativeClustering

ac=AgglomerativeClustering(n_clusters=4,affinity="euclidean",linkage="ward") 

tahmin=ac.fit_predict(x)



plt.scatter(x[tahmin==0,0],x[tahmin==0,1],color="r",s=100)

plt.scatter(x[tahmin==1,0],x[tahmin==1,1],color="g",s=100)

plt.scatter(x[tahmin==2,0],x[tahmin==2,1],color="b",s=100)

plt.scatter(x[tahmin==3,0],x[tahmin==3,1],color="y",s=100)

plt.title("AgglomerativeClustering")

plt.show()

#Burada da dendrogram ile Kaç küme oluşacağını tahmin edebiliriz.
import scipy.cluster.hierarchy as sch

den=sch.dendrogram(sch.linkage(x,method="ward"))

plt.show()

# Bu şekilde dendrogram çizilebilir ve yorum yapılabilir. 

# Görüntü büyüdüğü zaman daha iyi anlaşılacaktır. Hangi kümelerin seçildiği vs.