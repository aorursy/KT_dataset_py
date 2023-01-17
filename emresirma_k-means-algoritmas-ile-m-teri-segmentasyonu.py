#kütüphanelerin kurulumu

import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt



#tüm sütunları ve satırların görüntülenmesi

pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', None);



#virgulden sonra gösterilecek olan sayı sayısı

pd.set_option('display.float_format', lambda x: '%.0f' % x)
#veri setini okuma

df = pd.read_csv("../input/online-retail-ii-uci/online_retail_II.csv")
#ilk 5 gözlemin seçimi

df.head() 
#en cok siparis edilen urunlerin sıralaması

df.groupby("Description").agg({"Quantity":"sum"}).sort_values("Quantity", ascending = False).head()
#toplam kaç fatura sayısı

df["Invoice"].nunique()
#en pahalı ürünler 

df.sort_values("Price", ascending = False).head()
#en fazla sipariş sayısına sahip ilk 5 ülke

df["Country"].value_counts().head()
#toplam harcamayı sütun olarak ekledik

df['TotalPrice'] = df['Price']*df['Quantity']
#hangi ülkeden ne kadar gelir elde edildi

df.groupby("Country").agg({"TotalPrice":"sum"}).sort_values("TotalPrice", ascending = False).head()
#en eski alışveriş tarihi

df["InvoiceDate"].min() 
#en yeni alışveriş tarihi

df["InvoiceDate"].max()
#değerlendirmenin daha kolay yapılabilmesi için bugünün tarihi olarak 1 Ocak 2012 tarihi belirlendi.  

today = pd.datetime(2012,1,1) 

today
#sipariş tarihinin veri tipinin değiştirilmesi

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
#0'dan büyük değerleri alınması, bu işlem değerlendirmeyi daha kolaylaştıracak

df = df[df['Quantity'] > 0]

df = df[df['TotalPrice'] > 0]
#eksik verilere sahip gözlem birimlerinin df üzerinden kaldırılması

df.dropna(inplace = True) 
#veri setinde eksik veri yok

df.isnull().sum(axis=0)
#boyut bilgisi

df.shape 
df.describe([0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95, 0.99]).T

#belirtilen yüzdelere karşılık gelen gözlem birimlerinin açıklayıcı istatistik değerleri

#değerlendirmeyi kolaylaştırması amacıyla df tablosunun transpozu alındı.
#ülkeye göre müşteri dağılımı

country_cust_data=df[['Country','Customer ID']].drop_duplicates()

country_cust_data.groupby(['Country'])['Customer ID'].aggregate('count').reset_index().sort_values('Customer ID', ascending=False)
#yalnızca Birleşik Krallık verilerini tutma

df_uk = df.query("Country=='United Kingdom'").reset_index(drop=True)

df_uk.head()
df.head()
df.info() 

#dataframe'in indeks tipleri, sütun türleri, boş olmayan değerler ve bellek kullanım bilgileri
#Recency ve Monetary değerlerinin bulunması

df_x = df.groupby('Customer ID').agg({'TotalPrice': lambda x: x.sum(), #monetary value

                                        'InvoiceDate': lambda x: (today - x.max()).days}) #recency value

#x.max()).days; müşterilerin son alışveriş tarihi
df_y = df.groupby(['Customer ID','Invoice']).agg({'TotalPrice': lambda x: x.sum()})

df_z = df_y.groupby('Customer ID').agg({'TotalPrice': lambda x: len(x)}) 

#kişi başına düşen frequency değerini bulunması
#RFM tablosunun oluşturulması

rfm_table= pd.merge(df_x,df_z, on='Customer ID')
#Sütun isimlerini belirlenmesi

rfm_table.rename(columns= {'InvoiceDate': 'Recency',

                          'TotalPrice_y': 'Frequency',

                          'TotalPrice_x': 'Monetary'}, inplace= True)
#RFM Tablosu

rfm_table.head()
#Recency için tanımlayıcı istatistikler

rfm_table.Recency.describe()
#Recency dağılım grafiği

import seaborn as sns

x = rfm_table['Recency']



ax = sns.distplot(x)
#Frequency için tanımlayıcı istatistikler

rfm_table.Frequency.describe()
#Frequency dağılım grafiği, 1000'den az Frequency değerine sahip gözlemlerin alınması

import seaborn as sns

x = rfm_table.query('Frequency < 1000')['Frequency']



ax = sns.distplot(x)
#Monetary için tanımlayıcı istatistikler

rfm_table.Monetary.describe()
#Monatary dağılım grafiği, 10000'den az Monetary değerine sahip gözlemlerin alınması

import seaborn as sns

x = rfm_table.query('Monetary < 10000')['Monetary']



ax = sns.distplot(x)
#çeyreklikler kullanılarak dört parçaya bölünme işlemi

quantiles = rfm_table.quantile(q=[0.25,0.5,0.75])

quantiles = quantiles.to_dict()
quantiles
#Dönüştürme işlemi

from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler((0,1))

x_scaled = min_max_scaler.fit_transform(rfm_table)

data_scaled = pd.DataFrame(x_scaled)

#burada değerlerimizi normalleştirdik
df[0:10]
plt.figure(figsize=(8,6))

wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++',n_init=10, max_iter = 300)

    kmeans.fit(data_scaled)

    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
kmeans = KMeans(n_clusters = 4, init='k-means++', n_init =10,max_iter = 300)

kmeans.fit(data_scaled)

cluster = kmeans.predict(data_scaled)

#init = 'k-means ++' bu daha hızlı çalışmasını sağlar
d_frame = pd.DataFrame(rfm_table)

d_frame['cluster_no'] = cluster

d_frame['cluster_no'].value_counts() # küme başına kişi sayısı (Custer ID numarası)
rfm_table.head()
#kümelerin ortalama değerleri

d_frame.groupby('cluster_no').mean()