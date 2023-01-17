# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Visualization
import seaborn as sns # Visualization

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/iris/Iris.csv")
data.head(10) # head içerisine yazdığımız sayı kadar ilk satırları bize verir. Eğer bir şey yazmazsak ilk 5 satırı verecektir.
data.tail() # tail ise satırları sondan başlayarak verir.
data["Species"].value_counts() # value_counts() bize belirttiğimiz sütundaki her bir değişkenin kaç adet olduğunu döner.
sns.countplot(data["Species"])
data.isnull().any() # Verimizde herhangi bir NaN değer var mı kontrol ettik.
data.drop(["Id"],axis=1,inplace=True) # Id bizim işimize yaramayacak olan kullanmayacağımız bir kısım. Bu nedenle id sütununu verimizden çıkarıyoruz.
data.head() # Artık id kısmı yok verimizde.
# Şimdi feature'larımızın ikili ilişkilerine bakalım
sns.pairplot(data,hue="Species")
plt.scatter(data["PetalLengthCm"],data["PetalWidthCm"],color="black")
plt.xlabel("PetalLength")
plt.ylabel("PetalWidth")
plt.show()
# Biz bu verileri aralarındaki benzerliklere göre ayırmadan önce bu böyle bir görünüşe sahip diye düşünebiliriz.
x=data.drop(["Species"],axis=1)
y=data["Species"]
x
y
# Wcss ile en optimal k değerini buluyoruz.

from sklearn.cluster import KMeans

wcss=[]

# Burada hangi k değerini seçmeliyiz onu belirliyoruz. k => number of cluster

for k in range(1,15):
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(x)
    
    wcss.append(kmeans.inertia_) # inertia_ her bir k değeri için wcss hesaplar.
    
# wcss bizim elbow yani dirsek kısmını bulmamızı sağlar.
# Yani ne zaman dirsek yaparsa o k değeri bu model için en iyisidir demektir.
    
plt.plot(range(1,15),wcss)
plt.xlabel("number of k value")
plt.ylabel("wcss")
plt.show()

# O halde k=3 için bakalım şimdi.

kmeans2=KMeans(n_clusters=3)
kmeans2.fit(x)
y_pred=kmeans2.predict(x) # Burada x verilerimize göre tahmin yapıyoruz.

plt.scatter(data.loc[:,"PetalLengthCm"],data.loc[:,"PetalWidthCm"],c=y_pred,cmap="rainbow") # Görselleştirme
plt.xlabel("PetalLength")
plt.ylabel("PetalWidth")
plt.show()
# Dendrogram
# Dendrogram çizdirmek için sci-py kütüphanesini kullanacağız

from scipy.cluster.hierarchy import linkage,dendrogram

merg=linkage(x,method="ward")
dendrogram(merg,leaf_rotation=90) # leaf_rotation 90 derece yapmamızı sağlıyor yani çizdirirken dik olmasını.
plt.xlabel("data points")
plt.ylabel("euclidian distance")
plt.show() 
from sklearn.cluster import AgglomerativeClustering

ac=AgglomerativeClustering(n_clusters=3,affinity="euclidean",linkage="ward")
y_prediction=ac.fit_predict(x) # fit_predict modeli eğitip tahmin yapmamızı sağlar.

plt.scatter(data.loc[:,"PetalLengthCm"],data.loc[:,"PetalWidthCm"],c=y_prediction,cmap="rainbow") # Görselleştirme
plt.xlabel("PetalLength")
plt.ylabel("PetalWidth")
plt.show()