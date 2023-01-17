#Öncelikle kütüphanleri tanımlayalım 

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns

#Şimdi datamızı okuyalım ve bu çalışmada United States csv datasından yararlanacağız. 

data=pd.read_csv("../input/youtube-new/USvideos.csv")

#Datamızın bilgilerine bakalım 

data.info()

#Görüldüğü üzere 40949 adet verimiz var biz bu çalışmada başlıkların hepsini incelemeyeceğiz 

#Korelasyona bakalım 

data.corr() 

#Görüldüğü üzere görüntülenme ile beğenilme oranı 1 e çok yakın yani doğru orantılı durumda 

#Şimdi görselleştirelim 

f,ax = plt.subplots(figsize=(8,8)) #Bu komut 8x8 lik subplot olusturdu

sns.heatmap(data.corr(),annot=True, linewidths=1, fmt= '.1f',ax=ax) #Bu görsel seaborn kütüphanesi ile yapıldı

plt.show()

#Şimdi hangi başlıklarla inceleme yapacağımızı görelim

data.columns

#Bir de ilk ve son 10 ar videomuzu görelim 

data.head(10)

#data.tail(10)
#Şimdi elimizdeki verileri Matplotlib kütüphanesi ile basitce oluşturalım (Izlenme ve beğenilme sayılarına göre)

data.views.plot(kind='line',color='red',label='Views',linewidth=0.5, alpha=1, grid=True, linestyle='-',figsize=(10,10))

data.likes.plot(kind='line',color='purple',label='Likes',linewidth=0.1, alpha=0.5, grid=True, linestyle='steps')

data.dislikes.plot(kind='line',color='g',label='Dislikes',linewidth=0.3, alpha=0.5, grid=True, linestyle=':')

plt.legend(loc='upper left')

plt.xlabel('Veri sayısı')

plt.ylabel('y axis')

plt.title('Views - Likes oranı')

plt.show()

#Görüldüğü üzere genellikle görüntülenme arttıkça beğenilme sayısı da artmıs durumda 



#Şimdi de scatter plot ile Likes ve Dislikes arasındaki ilişkiyi inceleyelim

data.plot(kind='scatter', x='likes',y='dislikes',alpha=0.7, linewidth=1, color='orange',figsize=(10,10))

plt.show()

#Birde histogram ile kaç tane youtube videosunun aynı kategoride olduğunu görelim

data.category_id.plot(kind='hist',color='r',bins=15, figsize=(10,6))

plt.xlabel('Kategori')

plt.ylabel('Frekansı')

plt.title('Kategorisel sınıflandırma')

plt.show()
#Şimdi pandas yardımı ile filtreleme yapalım ve genel görüntülenme ortalamasını gördükten sonra bir filtreleme yapalım

ort= data.views.mean()

#Görüldüğü üzere ortalama izlenme 2 360 784.7 şimdi bir filtreleme ile ortalamanın üzerindeki video sayısını ve hangileri olduğunu görelim 

x= data['views']>ort

print(x.sum())

data[x]



#Burada comment_count baz alınarak List Comp uygulaması yapalım 

ort_yorum=sum(data.comment_count)/len(data.comment_count)

print('ortalama yorum sayisi:',ort_yorum)

data['populerlik(yorumsal)']=['Populer' if i >ort_yorum else 'Pop degil' for i in data.comment_count]

data.loc[:10,['title','populerlik(yorumsal)','comment_count']]

data.columns

#Öncelikle data ile ilgili bir sorun var mı bakalım örneğin NaN veri varsa yok edelim

data.info()

#Herşey null object o zaman box ile plotlayalım
#data.boxplot(column='title',by='dislikes')

#plt.show()
#Şimdi melt komutu ile datanın belli featurelerini çekelim

melted_data=pd.melt(frame=data,id_vars=['title','views'],value_vars=['likes'])

melted_data
data1=data.head()

data2=data.tail()

conc_data_row=pd.concat([data1,data2],axis=0,ignore_index=True) #Dikey olarak birleştirdik,yatay için axis=1 yapacaktık

conc_data_row
data.dtypes
data.plot(subplots=True)

plt.show()
#Şimdi pandas ile Time Series oluşturalım 

#trending date burda bizim time seriemizin indexi olabilir ama önce trending datein yapısı seri onu bi str yapalım olan publish time timeseries yapalım

# datetime_object= pd.to_datetime(data.trending_date) #Şeklinde ama burada tarihler genelde aynı ve yıl ay gün formatında yazılmamıs

#index içinde data=data.set_index("datetime_object")



# data=data.set_index("bir sütun ismi") yapılabilir

data["category_id"][1]
data.index.name = "index_name"

data.head()
data3=data.copy()

data3=data.set_index(["channel_title","title"])

data3