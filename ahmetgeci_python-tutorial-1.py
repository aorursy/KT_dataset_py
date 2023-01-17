#Değişkenleri tanımla

mesaj = "Merhaba Dünya"

plaka = 35

boy = 1.85

veriBiliminiSeviyorum = ( 1==True )
#mesaj değişken değerini yazdır

print("Mesaj: ", mesaj)

#veriBiliminiSeviyorum değişkenini yazdır

print(veriBiliminiSeviyorum)

#Tek satırda birden fazla değişkene değer atama

plaka, boy  = 35, 1.85
print(type(mesaj), type(plaka), type(boy), type(veriBiliminiSeviyorum))

renkListesi = ["sarı","kırmızı", "siyah", "beyaz", "bordo", "mavi"]
renkListesi[0] 
renkListesi[5] 
renkListesi[-1] 
renkListesi[-3] 
renkListesi[::] 
renkListesi[1::1] 
renkListesi[1::2]
renkListesi[:4] 
renkListesi[::-1] 
renkListesi[:2:-1] 
renkListesi[0:2] = ['sarı', 'lacivert'] 
renkListesi
renkListesi.append('yeşil')

renkListesi 
renkListesi.remove('siyah')

renkListesi 
renkListesi = renkListesi + ['turuncu']

renkListesi 
#liste metotları
renkListesi.reverse()

renkListesi
renkListesi.sort()

renkListesi 
#sözlük kullanımı

notlar = {

    "0001-Ada": 83,

    "0002-Cem": 79,

    "0003-Sibel" : 82 

}



#sözlük elemanına erişim

notlar["0001-Ada"] #83
#sözlüğe yeni eleman ekle

notlar["0004-Nil"] = 99

notlar 
#sözlükteki elemanı sil

del(notlar["0001-Ada"])

notlar 
("0004-Nil" in notlar) #True
#sözlük metotları
notlar.keys()
notlar.items() 
notlar.values() #dict_values([79, 82, 99])
#Referans tür örneği (yöntem 1)

sayilar = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#yanlış bir liste kopyalama yöntemi

kopya_sayilar = sayilar
#aşağıdaki değişiklik sadece kopya_sayilar nesnesinde mi yapılıyor?

kopya_sayilar[0] = 99
#her iki nesnenin de 0 indisli elemanı değişti

print("sayilar:", sayilar) 

print("kopya_sayilar:",kopya_sayilar) 
#liste kopyalama için diğer yöntem (yöntem 2)

sayilar = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

#legal kopyalama yöntemi

kopya_sayilar = sayilar[::] #kopya_sayilar = list(sayilar)
#her iki nesnenin de 0 indisli elemanı değişiyor mu?

kopya_sayilar[0] = 99

print("sayilar:", sayilar) 

print("kopya_sayilar:",kopya_sayilar) 
#fonksiyon tanimi

def kabarcikSiralama(dizi):

    eleman_sayisi = len(dizi)

    #Tüm elemanları dön

    for i in range(eleman_sayisi):

        for j in range(0, eleman_sayisi - i - 1):

            #Yer değiştirme

            if dizi[j] > dizi[j+1] :

                dizi[j], dizi[j+1] = dizi[j+1], dizi[j]
#fonksiyon kullanimi

sayilar = [3, 12, 9, 4, 5, 8, 11, 14, 2, 1]
kabarcikSiralama(sayilar)

for i in range(len(sayilar)):

    print ("%d" %sayilar[i]), 
#fonksiyon kullanımına örnek

def basitModel(model):

    model.add(Dense(21, input_dim=21, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal'))

    return model



def derinModel(model):

    model.add(Dense(21, input_dim=21, kernel_initializer='normal', activation='relu'))

    model.add(Dense(10, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal'))

    return model



model = Sequential()

model = basitModel(model)

model.compile(loss='mean_squared_error', optimizer='adam')
class BaseClassifier(object):

    def __init__(self):

        pass

    def get_name(self):

        raise NotImplementedError()

    def fit(self, x, y):

        raise NotImplementedError()

    def predict(self, x):

        raise NotImplementedError()
class BaseTree(BaseClassifier):

    def __init__(self, maks_derinlik):

        super().__init__()

        self.height = height



    def get_name(self):

        return "Base Tree";

 

    def fit(self, x, y):

        self.X = x

        self.Y = y

 

    def predict(self, x):

        y = np.zeros((x.shape[0],))

        for i in range(x.shape[0]):

            y[i] = self.__predict_single(x[i, :])

        return y
import numpy as np



#Dizi oluşturma

d1 = np.array([5.5,9,10])

d2 = np.array([(3.5,8,11), (4,7,9), (2,2,1.1)], dtype=float)



#Fark alma 1. Yöntem

d3 = d2 - d1

print("Fark 1 / d3 --> ", d3)



#Fark alma 2. Yöntem

d3 = np.subtract(d1, d2) 

print("Fark 2 / d3 --> ", d3)



#d1 ve d2'yi toplayıp d1 üzerine yazma

d1 = d1 + d2

print("Toplam d1 --> ", d1)

d1



#Değeri 3'ten büyük elemanların indislerini bul

sonuc = d1 > 9

print(sonuc)





#Bulunan indisleri kullanarak, elemanları ekranra yazdır

print("3'ten büyük elemanlar -->", d1[sonuc])
#İki matrisin çarpımı

import numpy as np 

d4 = np.dot(d1, d2)

print("Çarpım d4:", d4)



#Matristen 1.sütunu çıkartma

d4 = np.delete(d4,0,1)

print("Çıkartma d4:", d4)
#2x5’lik sıfır matrisi yaratma

SifirMatrisi = np.zeros([2,5])
#Dizideki en küçük eleman bulma

print("d4 min:", np.min(d4))



#Dizideki en büyük eleman bulma

print("d4 max:", np.max(d4))



#Dizinin ortalamasını alma

print("d4 ortalama:", d4.mean())



#Dizinin toplamını bulma

print("d4 toplam:", d4.sum())



#Karekök alma

print("d4 karekök-->", np.sqrt(d4))



#Dizinin logaritmasını hesaplama

print("d4 logaritma-->", np.log(d4))



#Tranpoz alma

print("d4 transpoz:", np.transpose(d4))
import pandas as pd



#Pandas Series tanımalama

s = pd.Series([11, -3, -1, 2], index=['a','b','c','d'])



#2 boyutlu veriyi Pandas DataFrame yapısına çevirme

#columns parametresi ilse dataframe veri yapısında sütun isimleri tanımlanır.

veriTablo = pd.DataFrame(veri, columns=['Ulke','Sehir','Nufus'])



#Csv dosyasından veri okuma

pd.read_csv('dosyaAdi.csv', nrows=7, sep=',')



#Tablodaki konumuna göre veri seçme

veriTablo.iloc([0],[0])



#Başlığa göre veri seçme

veriTablo.loc([5,['Nufus']])



#Veri silme

s.drop(['a','c'])



#Tablo hakkında basit bilgiler edinme

df.info() #tablonun satır, sütün sayısı, veri tipi, eksik veri sayısı gibi bilgileri verir.

veriTablo.shape() #tabloda bulunan satır ve sütun sayısını verir

veriTablo.describe()#tablonun genel tanımlayıcı istatistik değerlerini verir

df.min() #her sütün için en küçük veriyi bulur

df.mean() #her sütün için ortalama değer hesaplar



#hata alırsınız çünkü bir dataseti tanımlı değildir !!!!!

list1 = [1, 2, 3, 4, 5, 6]

list2 = [i/2 for i in list1]

print(list2) 
x, y = 5, 5

if (x > y):

    print("x > y")

elif (y > x):

    print("y > x")

else:

    print("y = x")
#while döngüsü

kosul, j = True, 0

while (kosul):

    print(j)

    j += 1

    kosul = (j != 5)





#for döngüsü

for i in range(0, 5):

    print(i)

#lambda fonksiyon 1

fnc = lambda x : x + 1

print(fnc(1)) #Çıktı: 2

print(fnc(fnc(1))) #Çıktı: 3



#lambda fonksiyon 2

fnc2 = lambda x, y : x + y

print(fnc2(4,7)) #Çıktı: 11

print(fnc2(4,fnc(1))) #Çıktı: 6
#lamba fonksiyon içeren fonksiyon tanımları

def fnc3(n):

  return lambda x : x ** n



fnc_kare_al = fnc3(2) #Dinamik kare alma fonksiyonu oluşturuluyor

fnc_kup_al = fnc3(3) #Dinamik küp alma fonksiyonu oluşturuluyor



print(fnc_kare_al(3))

print(fnc_kup_al(3))
import pandas as pd

data = [

        ['D1', 'Sunny','Hot', 'High', 'Weak', 'No'],

        ['D2', 'Sunny','Hot', 'High', 'Strong', 'No'],

        ['D3', 'Overcast','Hot', 'High', 'Weak', 'Yes'],

        ['D4', 'Rain','Mild', 'High', 'Weak', 'Yes'],

        ['D5', 'Rain','Cool', 'Normal', 'Weak', 'Yes'],

        ['D6', 'Rain','Cool', 'Normal', 'Strong', 'No'],

        ['D7', 'Overcast','Cool', 'Normal', 'Strong', 'Yes'],

        ['D8', 'Sunny','Mild', 'High', 'Weak', 'Yes'],

        ['D9', 'Sunny','Cool', 'Normal', 'Weak', 'No'],

        ['D10', 'Rain','Mild', 'Normal', 'Weak', 'Yes'],

        ['D11', 'Sunny','Mild', 'Normal', 'Strong', 'Yes'],

        ['D12', 'Overcast','Mild', 'High', 'Strong', 'No'],

        ['D13', 'Overcast','Hot', 'Normal', 'Weak', 'Yes'],

        ['D14', 'Rain','Mild', 'High', 'Strong', 'No'],

       ]

df = pd.DataFrame(data,columns=['day', 'outlook', 'temp', 'humidity', 'windy', 'play'])

df
df.info()
df.min()
df.max()
df.shape
df.describe()
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder() 

df['outlook'] = lb.fit_transform(df['outlook']) 

df['temp'] = lb.fit_transform(df['temp'] ) 

df['humidity'] = lb.fit_transform(df['humidity'] ) 

df['windy'] = lb.fit_transform(df['windy'] )   

df['play'] = lb.fit_transform(df['play'] ) 
df
df.describe()
X = df.iloc[:,1:5] 

Y = df.iloc[:,5]
X
Y