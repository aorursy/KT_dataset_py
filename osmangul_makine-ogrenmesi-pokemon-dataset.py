#KISALTMALAR VE ANLAMLARI:
# Total: Bir pokemonun tüm savaş istatistiklerinin toplamıdır.
# HP : Pokemonun sağlık derecesi.Ne kadar büyükse o kadar iyi
#Attack : Pokemonun normal fiziksel saldırısı.Ne kadar büyükse o kadar çok fiziksel hasar verir.
#Defense: Pokemonun fiziksel saldırılardan korunma gücü ne kadar yüksekse o kadar çok korunur.,
#sp_Atack: Pokemonun yetenek saldırısı
#Sp_Def. Pokémon'un temel özel savunması. Ne kadar büyükse, özel yetenek ile saldırı olduğunda kendini o kadar savunabilir
#isLegendery : Pokémon'un efsanevi olup olmadığını gösteren Boolean. Efsanevi Pokémon,daha güçlü, benzersiz yeteneklere sahip ve bulması gerçekten zor olan pokemon.
#Speed : Pokemonun saldırı hızı.Yüksek olursa daha hızlı vuruyor.
#Generation : Pokemonun piyasaya sürüldüğü nesil.1-6 arası bir sayıdır.
#hasGender: Pokemon erkek ya da dişi olarak sınıflandırılabilir olduğunu gösteren boolean.
#Pokémon'un cinsiyeti olması durumunda, erkek olma olasılığını verir. Dişi olmanın olasılığı da -1 kabul edilir.Bu değişken sayısal ve ayrıktır çünkü Pokémon'un doğada kadın veya erkek olarak görünme olasılığı olmasına rağmen, sadece 7 değer vardır: 0, 0.125, 0.25, 0.5, 0.75, 0.875 ve 1.
#hasMegaEvolution: Bir pokemonun mega evrim geçirip geçirmediğini gösteren boolean.Dönüşüm varsa çok daha güçlü bir pokemon ortaya çıkar
# Height_m: Pokemonun yüksekliği metre cinsinden variable.
# Weight_kg : Pokemonun ağırlığı kilogram cinsinden
# Catch_Rate: Pokemon yakalama oranı.Pokemon yakalamanın ne kadar kolay olduğunu gösteren derecelendirme.3-255 arası değer alır.


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# pokemon_alopez247.csv dosyamızı pandas kütüphanesi ile data değişkenine atama yaptık
data = pd.read_csv("../input/pokemon_alopez247.csv")

data.describe() # Veri kümemize ait basit istatistikleri veriyor.
data.info() # Veri kümesi ile alakalı bellek kullanımı ve veri istatistikleri

data.head() # Dataset tablosunun ilk 5 bilgisini veriyor..
data.tail() # Son 5 satırı veriyor...
data.shape #Shape bir özelliktir.Method olmadığı için parantezle yazmadık.Kaç satır ve sütundan oluştuğunu belirtiyor.
# matplot kutuphanesini importla
import matplotlib.pyplot as plt

# Defense sütununu sec ve histogramını çıkar.
plt.hist(data['Defense'])
# Histogramın başlığı
plt.title('Defense İçin Histogram Bilgisi')
# Histogramı göster
plt.show()
#Grafik üzerinde de görüldüğü gibi 50-100 aralıgında Defense özelliği yüksek seviye de seyir izliyor.
# Kutu çizim grafiği incelemesi
data.plot(kind='box', sharex=False, sharey=False)

# Değerlerin dağılımının kutu grafiği üzerinde gösterimini bu şekilde rahatça gözlemeyebiliriz.
#(korelasyon: iki rassal değişken arasındaki doğrusal ilişkinin yönünü ve gücünü belirtir)
#Düz metin halinde korelasyon gösterimi
import seaborn as sns
corr=data.corr()
data.corr()
#Korelasyon değeri 1'e yaklaştıkça ilişkinin iyi olduğunu gösterir.Zıttı durumda da anlaşılacağı üzere bağlılık azalır
#Korelasyonu yüksek ilişki tablodaki 'Attack' ve 'Total' arasındaki ilişki;
#Korelasyonu düşük olan ilişki ise 'Catch_Rate' ile 'Speed'ilişkisi...
#KORELASYON SEABORN HEATMAP (ISI HARİTASI) GÖSTERİM
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values
           )
#Korelasyonları yüksek olan Total ve Sp_Atk öznitelikleri için plotting 
import matplotlib.pyplot as plt
data.plot(x='Total', y='Sp_Atk', style='go')  #o yuvarlak şekil g de green = go
plt.title('TOTAL - Sp_Atk İlişki Tablosu')  #Tablonun başlığını belirleyen kısım
plt.xlabel('Total')  #tablonun x ekseninine başlık ekleme
plt.ylabel('Sp_Atk') #tablonun y eksenine başlık ekleme
plt.show() 
#Özel atak gücü 25-70 değerleri arasında iken 300 'total' derecesi bir hayli yoğun gözükmekte.Yani 'Sp_Atk' özelliği düşük olanların 'Total' özelliği de düşük diyebilir aralarında doğru orantı olduğunu gözlemleyebiliriz... 
# --2.KISIM VERİ ÖN İŞLEME-- 

#EKSİK DEĞER KONTROLÜ (Null olan değerleri buuyoruz)
totalNull=data.isnull().sum()

# Toplam boş sayısı
totalNull.sum()

#Eksik değer tablosu
def eksik_deger_tablosu(data): 
    mis_val = data.isnull().sum()
    mis_val_percent = 100 * data.isnull().sum()/len(data)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Eksik Değerler', 1 : '% Değeri'})
    return mis_val_table_ren_columns
eksik_deger_tablosu(data)
#TABODA BOŞ OLAN YERLERE 'boş' YAZARAK BU ŞEKİLDE DOLDURABİLİRİZ.
data['Type_2'] = data['Type_2'].fillna('boş')
data['Pr_Male'] = data['Pr_Male'].fillna('boş')
data['Egg_Group_2'] = data['Egg_Group_2'].fillna('boş')
#Doldurduktan sonra da tabloyu kontrol edelim.
data
#UÇ DEĞERLERİN BULUNMASI
#1. Max değer...
data.max()
#UÇ DEĞERLERİN BULUNMASI
#1. Min değer...
data.min()
#2.Aykırı Değer Tespiti
import seaborn as sns
sns.boxplot(x=data['Total'])

#Aykırı değerlerin tespiti,belirli bir veri kümesini ele alarak bize alt çeyrek,üst çeyrek,ortalama değer ve sınırların dışında kalan değerler konusunda bilgi verir.
#Bu grafikte de alt ve üst açıklıklar belirlenmiş ortalama ise boxplot ile çizilen kutu grafiğinin içindeki çizgi ile belirtilmiş.

#Yeni öznitelik ekleme( Pokemon toplam saldırı gücü (özel ve fiziksel))
topAttack = (data['Attack']+data['Sp_Atk'])
data['TotalAttackPower'] = topAttack
data
# Veri Normalleştirme ( Oluşturduğumuz özniteliğe normalleştirme yaptık)
from sklearn import preprocessing

# TotalGoalFT özniteliğini normalleştirmek istiyoruz
x = data[['TotalAttackPower']].values.astype(float)

#Normalleştirme için MinMax methodu kullandık.
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
data['NormallestirilenVeri'] = pd.DataFrame(x_scaled)

data
#Tüm değerlerini sayısal yapma ( Label Encoder ile )
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()

NewData = data[::] # data veri kumesinin içeriğini NewData isimli yeni verikumesine kopyaladık...  
NewData['Name'] = lb.fit_transform(NewData['Name'])
NewData['isLegendary'] = lb.fit_transform(NewData['isLegendary'])


# NewData veri kumesinin son hali...
NewData
# ---3.KISIM : MODEL EĞİTİMİ ---

#LinearRegresyon Modeli  --> Model 1
#Kolerasyonu düşük olan 'Catch_Rate' ile 'Total' arasındaki ilikişkiyi Linear Regresyon ile inceleyelim
data.plot(x='Catch_Rate', y='Total', style='ro')  
plt.title('Catch_Rate - Total')  
plt.xlabel('Catch_Rate')  
plt.ylabel('Total')  
plt.show() 
#Grafikten de anladığımız gibi doğrusal ya da düzgün bir oluşum göstermiyor bize.Catch_Rate özelliği 3-255 arası değer alan bir öznitelik.
#Her ne kadar grafikte 3 ve 255 değerleri gösterilmemiş olsa da bunu biliyoruz.
#Doğrusal bir artış ya da azalış söz konusu değil bu açıkça ortada.
#İki öz nitelik korelasyon bağlamında birbiriyle düşük ilişki içerisinde.
#Yani bu iki özelliğin artıp azalması birbiri ile ilişkilendirme açısından yetersiz...
#Bir başka grafii daha incelemek gerekirse kendi oluşturduğumuz 'TotalAttackPower' özelliği ile 'Total' özelliği arasındaki ilişkiyi gözlemleyelim.
data.plot(x='TotalAttackPower', y='Total', style='go')  
plt.title('TotalAttackPower - Total')  
plt.xlabel('TotalAttackPower')  
plt.ylabel('Total')  
plt.show() 
#Görüldüğü üzere tablo doğrusal çizilmiş bir sıra üzerine dizilmiş veriler gibi gözüküyor.
#Yani iki veri grubu arasında doğrusal artış gösteren bir ilişki var diyebiliriz...
#Verilerin değişimleri birbiri üzerinde etki edecek değişimler olabilir biri artarkan diğeri artabilir ve bu durum tersinir olabilir.
#İncelediğimiz yeşil pointler ile gösterilmiş grafiğide göz önünde bulundururak model eğitimi yapmaya başlayabiliriz...
# Eğitim için ilgili öznitelik değerlerini seçimi
Y = data.iloc[0:, 4:5].values  #Total Değerlerinin Tutulduğu Sütun
X = data.iloc[0:, 5:6].values  #TotalPowerAttack Değerlerinin Tutulduğu Sütun
Y
X
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0) # Test verisi yuzde 20 ile belirlenecek.
#Linear Regression modeli için kendi adı olan LinearRegression kütüphanesini işleyerek model geliştireceğiz.
from sklearn.linear_model import LinearRegression 
#model değişkenimizi LinearRegression fonksiyonuna atama işlemi yapıyoruz..
model = LinearRegression()
#.fit modeli eğitmek için kullanılan fonksiyon
model.fit(X_train, y_train)  
#Grafiğin kesim noktalasını ve eğim özelliğini görmek için aşağıdaki komutları kullanıyoruz...
print("Kesim Noktası:", model.intercept_) 
print("Doğrunun Eğimi:", model.coef_) 
X_test # elde edilen x verilerini görelim... 
y_test #elde edilen y verilerini görelim
y_tahminDeger = model.predict(X_test)  # y_tahminDeger ML sonucunda modelin tahmin ile ürettiği değerler...
#matplotlib kütüphanesi aracılığı ile grafiğimizi oluşturuyoruz.Renk özelliklerini ve grafiğin başlıklarını belirliyoruz...
plt.scatter(X_train, y_train, color = 'red')
modelin_tahmin_ettigi_y = model.predict(X_train)
plt.plot(X_train, modelin_tahmin_ettigi_y, color = 'blue')
plt.title('TotalAttackPower - Total')
plt.xlabel('TotalAttackPower')
plt.ylabel('Total')
plt.show()
#Grafiği yorumlamak gerekirse gördüğümüz üzere doğrsusal bir sonuç üretildi.Bu doğrusal sonuç bizim reel verilerimizin olduğu
#kısımlar doğrultuda seyir gösteriyor.Her ne kadar doğrultu dışında aykırı veriler olsa da grafik genelinde tahmin çizgisi
#ortalama doğru veriyor.
from sklearn import metrics   
# Hata Oranı Metrikleri Bulunan değerin düşük olması bekleniyor yani modeli geliştirdikçe minumuma gitmeli...
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_tahminDeger))  
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_tahminDeger))) 
#MSE değeri gözlemlenilmiş veriler göz önüne alınarak tahmin etme amacı güden bir prosedürdür.Ortalama kare hatası anlamına gelir.
#Bizim grafiğimizde MSE değeri çok yüksek çıktı bu bağlamda iyi bir sonuç olmadığını söyleyebilirim.
#Grafikteki aynı doğrultulu değerler olmasına rağmen aykırı değerler ve doğrultu dışı sapan değerler MSE değerinin yükselmesine sebep olmuştur.

#--------

#RMSE değeri de düşük olmalıdır.Büyük hataların ölçülmesi için kullanılan bir prosedürdür.Ancak bu modelde düşük değildir.
#Bu da modelimizin kötü iyi eğitilmediğini göstermektedir.
#Naive Bayes modelinin oluşturulması
from sklearn.metrics import confusion_matrix, classification_report #conf. matrix ve recall,procesion gibi değerleri görmek için kullanılır
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB #Naive Bayes modelini oluşturmak için GaussianNB kütüphanesi import edilir
from sklearn.model_selection import KFold 

kfold = KFold(3, True, 1)
#Eğitim ve doğrulama veri kümelerinin ayrıştırılması
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
# Gaussian Naive Bayes Modelini kullanarak model geliştirme yapıyoruz
model = GaussianNB()
# Seçili oznitelikler ile test degerlerini atadık...
scoring = 'accuracy'
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)
# test_size değerini 0.3 yaparak yüzde 30 oranında gösterilmesini sağladık

# Naive Bayes Modelinin eğitimi için .fit fonksiyonunu kullandık
model = model.fit(X_train, Y_train)
# tahmin etmek için .predict kullanıyoruz.
Y_tahmin = model.predict(X_test)

#sonuç değerlerinin matrix olarak gösterilmesi için cv_results kullandık
cv_results
#sonuç değerlerinin ortalamasını veriyor
msg = "%f (%f)" % (cv_results.mean(), cv_results.std())
msg
# RECALL ve Precision değerlerini almak için kullanılıyor ama hata alıyorum...
#print(classification_report(y_test, y_tahminDeger))
#print(confusion_matrix(y_test, y_tahminDeger))
# Acc doğruluk değerini bulmuş oluyoruz.Sayıyı 100 ile çarparak daha verimli bir sonuç elde ediyoruz...
print(" ACC: %%%.2f" % (metrics.accuracy_score(Y_test, Y_tahmin)*100))
#ACC değerinin yüksek olması gerekiyor ancak model iyi eğitilemediyse yeterince yüksek değer geri döndürmüyor.Bizim ACC sonucumuz 100'de 6,45 yani iyi bir sonuç değil.
#Model bu veri kümesi için yeterince iyi değil ya da yeterince iyi eğitilmemiş olabilir.
