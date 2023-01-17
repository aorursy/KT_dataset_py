import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import os
print(os.listdir("../input"))#implemente edilen sınıfların tanımları
data = pd.read_csv('../input/turkiye-student-evaluation_generic.csv')
df = pd.DataFrame(data)#kullanılan datasetin tanımı
df.head()#verisetinin başından değerler
df.tail()#verisetinin sonundan değerler
df.info()#verisetindeki öznieliklerle ilgili bilgiler
df.shape#verisetindeki örnek ve öznitelik sayıları
df.sample(25)#verisetinden rastgele 25 örnek
df.describe()#verisetinin temel istatistik değerleri
plt.figure(figsize=(15,6))
sns.countplot(x='class', data=df)#class özniteliğinin sahip olduğu değerlere göre örnek sayıları
plt.title("Histogram")
plt.hist(df["class"], bins = 13)
plt.show()#class özniteliğinin sahip olduğu değerlere göre örnek sayıları
plt.title("Histogram")
plt.hist(df["difficulty"], bins = 5, color = 'red')
plt.show()#difficulty özniteliğinin sahip olduğu değerlere göre örnek sayıları
df.corr()#korelasyon matrisi
f,ax=plt.subplots(figsize=(20,20))
sns.heatmap(df.corr(),annot=True, linewidth=.8,fmt='.2f',ax=ax)
plt.show()#verisetinin ısı haritası
plt.figure(figsize=(20, 20))
sns.boxplot(data=df.iloc[:,25:27]);#+ korelasyonları en yüksek olan Q21 ve Q22 özniteliklerinin ikili grafiği
df.isnull().sum()#özniteliklerde boş değer kontrolü
plt.figure(figsize=(20, 20))
sns.boxplot(data=df.iloc[:,0:32]);#uç değer grafiği
df.mean(axis=0,skipna=True)#özniteliklerin ortalamaları
df.median()#özniteliklerin ortancaları
df.std()#özniteliklerin standart sapmaları
df.mode()#özniteliklerin modları
dfq = df.iloc[:,5:33]
dfq.head()#Q1-Q28 arası öğrencilere yapılan anket sorularının ayrılması
from sklearn import preprocessing
x = dfq.values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
dfq = pd.DataFrame(x_scaled)
dfq.sample(25)#normalizasyon işlemi ve rastgele 25 örnek
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
dataset_questions_pca = pca.fit_transform(dfq)
dataset_questions_pca.shape#PCA uygulanarak verileri önişleme, 28 özniteliği 2 özniteliğe dönüştürme
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 7):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(dataset_questions_pca)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 7), wcss)
plt.title('Dirsek Metodu')
plt.xlabel('Küme Sayısı')
plt.ylabel('WCSS')#within-cluster sum of squares değeri
plt.show()#varsayılan 7 küme üzerinden dirsek noktasını bulmak için grafik çizimi
kmeans = KMeans(n_clusters = 3, init = 'k-means++')
y_kmeans = kmeans.fit_predict(dataset_questions_pca)#dirsek noktası 3'e göre verisetini tahminleme
plt.scatter(dataset_questions_pca[y_kmeans == 0, 0], dataset_questions_pca[y_kmeans == 0, 1], s = 100, c = 'yellow', label = 'Cluster 1')
plt.scatter(dataset_questions_pca[y_kmeans == 1, 0], dataset_questions_pca[y_kmeans == 1, 1], s = 100, c = 'green', label = 'Cluster 2')
plt.scatter(dataset_questions_pca[y_kmeans == 2, 0], dataset_questions_pca[y_kmeans == 2, 1], s = 100, c = 'red', label = 'Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'blue', label = 'Centroids')
plt.title('Öğrenci Kümeleri')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.show()#2 öznitelik(boyut) uyarınca oluşan kümeler ve merkezleri
from sklearn.cluster import AgglomerativeClustering
agglomerative = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
y_agglomerative = agglomerative.fit_predict(dataset_questions_pca)
X = dataset_questions_pca
plt.scatter(X[y_agglomerative == 0, 0], X[y_agglomerative == 0, 1], s = 100, c = 'yellow', label = 'Cluster 1')
plt.scatter(X[y_agglomerative == 1, 0], X[y_agglomerative == 1, 1], s = 100, c = 'green', label = 'Cluster 2')
plt.scatter(X[y_agglomerative == 2, 0], X[y_agglomerative == 2, 1], s = 100, c = 'red', label = 'Cluster 3')
plt.title('Öğrenci Kümeleri')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.show()#2 öznitelik(boyut) uyarınca Agglomerative metoda göre oluşan kümeler ve merkezleri
from sklearn.cluster import SpectralClustering
spectral = SpectralClustering(n_clusters=3, random_state=1, affinity='nearest_neighbors')
y_spectral = spectral.fit_predict(dataset_questions_pca)
X = dataset_questions_pca
plt.scatter(X[y_spectral == 0, 0], X[y_spectral == 0, 1], s = 100, c = 'yellow', label = 'Cluster 1')
plt.scatter(X[y_spectral == 1, 0], X[y_spectral == 1, 1], s = 100, c = 'green', label = 'Cluster 2')
plt.scatter(X[y_spectral == 2, 0], X[y_spectral == 2, 1], s = 100, c = 'red', label = 'Cluster 3')
plt.title('Öğrenci Kümeleri')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.show()#2 öznitelik(boyut) uyarınca Spectral metoda göre oluşan kümeler ve merkezleri
import collections
collections.Counter(y_kmeans)
#2358 başarısız, 2222 başarılı ve 1240 orta değerli öğrencinin yer aldığı küme meydana geldi
collections.Counter(y_agglomerative)
#810 başarısız, 2692 başarılı ve 2318 orta değerli öğrencinin yer aldığı küme meydana geldi
collections.Counter(y_spectral)
#601 başarısız, 4551 başarılı ve 668 orta değerli öğrencinin yer aldığı küme meydana geldi