# csv dosyalarını okumak için 

#pandası import ettik

import pandas as pd
# csv dosyamızı okuduk.

data = pd.read_csv('../input/iris/Iris.csv')
# Veriler düzenledik

v = data.iloc[:,1:-1].values
# KMeans sınıfını import ettik

from sklearn.cluster import KMeans
# KMeans sınıfından bir nesne ürettik

# n_clusters = Ayıracağımız küme sayısı

# init = Başlangıç noktalarının belirlenmesi

km = KMeans(n_clusters=3, init='k-means++',random_state=0)

# Kümeleme işlemi yaptık



km.fit(v)
# Tahmin işlemi yapıyoruz.

predict = km.predict(v)
# Küme merkez noktalarını ayarladık.

print(km.cluster_centers_)
# Grafik şeklinde ekrana basmak için matplotlib kütüphanesini import ettim 

import matplotlib.pyplot as plt

#Sonra kümeleme ve renk verilerini ayarladım



plt.scatter(v[predict==0,0],v[predict==0,1],s=50,color='red')

plt.scatter(v[predict==1,0],v[predict==1,1],s=50,color='blue')

plt.scatter(v[predict==2,0],v[predict==2,1],s=50,color='green')

plt.title('K-Means Iris Dataset')

plt.show()
#Sonra kümeleme ve renk verilerini ayarladım



plt.scatter(v[predict==0,0],v[predict==0,1],s=50,color='red')

plt.scatter(v[predict==1,0],v[predict==1,1],s=50,color='blue')

plt.scatter(v[predict==2,0],v[predict==2,1],s=50,color='green')

#burada siyah rengi ekledim çıktılara bakalım

plt.scatter(v[predict==3,0],v[predict==3,2],s=50,color='black')

plt.title('K-Means Iris Dataset')

plt.show()
seri = pd.Series([121,200,150,99])



seri.values