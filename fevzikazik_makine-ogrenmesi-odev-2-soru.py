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
# pokemon.csv dosyamızı pandas yardımıyla df değişkenine ata
df = pd.read_csv("../input/Pokemon.csv")
df
df.head() #ilk 5 satır
df.tail() #son 5 satır      
df.shape #satır ve sütun sayısı
df.info() #bellek kullanımı ve veri türleri
df.describe() #basit istatistikler
# matplot kutuphanesini importla
import matplotlib.pyplot as plt

# Attack sütununu sec ve histogramını çıkar.
plt.hist(df['Attack'])
# Başlıgını belirle
plt.title('Attack İçin Histogram Bilgisi')
# Histogramı göster
plt.show()

# Tüm kayıtların 'Attack' sütünu dikkate alındığında
# kayıtların daha çok 50 - 100 değerleri arasında daha çok olduğunu görünüyor...
# Kutu çizim grafiği incelemesi
df.plot(kind='box', sharex=False, sharey=False)

# Değerlerin Genel Olarak Hangi aralıklarda daha yoğun olduğunun görüntüsü...
# Korelasyon Gösterim
df.corr()

# Korelasyonuna baktığımızda 'Attack' ile 'Generation' özniteliklerinin korelasyonu gayet dusuk yani birbiriyle bağlılıkları düşük demek oluyor...
# ve başka bir örnek verecek olursak 'Sp. Atk' ile 'Total' öznitelikleri arasında korelasyon yüksek yani birbirine olan bağlılıkları yüksek gibi...
# Korelasyon Gösterim (matplotlib)
import matplotlib.pyplot as plt
plt.matshow(df.corr())
# Burdada özniteliklerin birbirlerine bağlılıklarının koyuluk derecesiyle gösterimi...
# Korelasyon Gösterim (seaborn) Isı haritalı gösterim
import seaborn as sns
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
# Renkler koyulaştıkça korelasyonun yüksek olduğu anlamına geliyor...
# Korelasyonları yüksek olan 2 tane öznitelik için plotting (çizim işlemi) 
df.plot(x='Attack', y='Speed', style='o')

plt.title('Attack - Speed')
plt.xlabel('Attack')
plt.ylabel('Speed')
plt.show()
# VERİ ÖN İŞLEME

#Eksik Değer Doldurma
#Null olan öznitelikleri buluyoruz
toplamBos = df.isnull().sum()

# Toplam boş sayısı
toplamBos.sum()
# Eksik değer tablosu
def eksik_deger_tablosu(df): 
    bos_deger = df.isnull().sum()
    bos_deger_yuzdesi = 100 * df.isnull().sum()/len(df)
    bos_deger_tablo = pd.concat([bos_deger, bos_deger_yuzdesi], axis=1)
    bos_deger_tablo_doldur_sutun = bos_deger_tablo.rename(
    columns = {0 : 'Eksik Değerler', 1 : '% Değeri'})
    return bos_deger_tablo_doldur_sutun

eksik_deger_tablosu(df)
# %70 üzerinde null değer içeren kolonları sil
tr = len(df) * .3
df.dropna(thresh = tr, axis = 1, inplace = True)
df
#Type 2 kolonundaki Null değerleri 'isimsiz' değeri ile doldur
df['Type 2'] = df['Type 2'].fillna('isimsiz')
df
# Aykırı Değer Tespiti
import seaborn as sns
sns.boxplot(x=df['Speed'])
# Burdaki değerler sonucunda ayrıkı değerimizi görebiliyoruz. 150-175 civarlarında görülüyor...
# Mevcut özniteliklerden yeni bir öznitelik oluşturma

# Apply fonksiyonu kullanarak çok hızlı olup olmadığını kontrol edip yeni kolon olarak ekle
def hiz_kontrolu(hiz):
    return (hiz >= 90)  # Hizlari 90 dan buyukse yeni kolondaki değeri true yapacak...

df['isVeryFast'] = df['Speed'].apply(hiz_kontrolu)
df
# Veri Normalleştirme
from sklearn import preprocessing

# Speed özniteliğini normalleştirmek istiyoruz
x = df[['Speed']].values.astype(float)

#Normalleştirme için MinMax normalleştirme yöntemini kullanıyoruz.
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df['Speed2'] = pd.DataFrame(x_scaled)

df
# Label Encoder ile Tüm metin değerlerini Sayisal yapma

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()

dfyeni = df[::] # df veri kumesinin içeriğini dfyeni isimli yeni verikumesine kopyaladık...  
dfyeni.drop(['#', 'Speed2'], axis=1, inplace = True) # Sutunların bazılarını sildik (Feature Extraction)

# Label Encoder ile metinleri Sayisallaştırdık...
dfyeni['Name'] = lb.fit_transform(dfyeni['Name'])
dfyeni['Type 1'] = lb.fit_transform(dfyeni['Type 1'])
dfyeni['Type 2'] = lb.fit_transform(dfyeni['Type 2'])
dfyeni['Legendary'] = lb.fit_transform(dfyeni['Legendary'])
dfyeni['isVeryFast'] = lb.fit_transform(dfyeni['isVeryFast'])

# dfyeni veri kumesinin son hali...
dfyeni
# Eğitim için ilgili öznitelik değerlerini seçimi (Feature Selection)
X = dfyeni.iloc[:,1:5] # 1 ile 5 arasında indexli sutunlardaki tüm satırları alir...
X
#Sınıflandırma öznitelik değerlerini seç
Y = dfyeni.iloc[:,5] # 5 indexli sutununa ait tüm satırları alir...
Y
# Naive Bayes Algoritması Modeli ile Makine Ogrenmesi... (Model 1)
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

# GaussianNB() Modelini kullandık...
model = GaussianNB()
# Sectiğimiz oznitelikleri  ile test degerlerini atadık...
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)
# burdaki test_size=0.3 demek yuzde 30 test verisi olacak demektir...

# Modeli fit yaptık yani öğretmek, eğitmek, beslemek anlamında vs...
model = model.fit(X_train, Y_train)
# .predict ile tahmin ettiriyoruz...
Y_pred = model.predict(X_test)

# Accuracy(Doğruluk)değeri... 
print(" ACC: %%%.2f" % (metrics.accuracy_score(Y_test, Y_pred)*100))
# LineerRegression ile Makine Ogrenmesi... (Model 2)

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# Test verimiz yuzde 20 olacak...


# Sklearn kutuphanesinden LineerRegression modelini sectik...
from sklearn.linear_model import LinearRegression  
model = LinearRegression()
# Modeli fit ediyoruz(eğitiyoruz)...
model.fit(X_train, y_train)

# Bazı değerleri...
print("Kesim noktası:", model.intercept_) 
print("Eğim:", model.coef_)

X_test
# Tahmin verileri
y_tahmin = model.predict(X_test)
y_tahmin
# Gerçek veriler ile Tahmin edilen verileri karşılaştırmak için geçici bir veri kumesi olusturduk...
df_karsilastirma = pd.DataFrame({'Gerçek': y_test, 'Tahmin Edilen': y_tahmin})  
df_karsilastirma

from sklearn import metrics   
# Hata Oranı Metrikleri Bulunan değerin düşük olması bekleniyor yani modeli geliştirdikçe minumuma gitmeli...
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_tahmin))  
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_tahmin))) 