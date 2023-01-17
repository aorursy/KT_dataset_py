import pandas as pd
import numpy as np
import matplotlib as plt
data = pd.read_csv('../input/data.csv',sep=";")
data=data.apply(lambda x:x.str.replace(",","."))
type(data.issizlik)
type(data.enflasyon)
type(data)
data.head(5)
data.tail()
data.tail(3)
data.columns
data.axes
data.shape
data.info()
data['enflasyon'] = data['enflasyon'].astype('float')

data['issizlik'] = data['issizlik'].astype('float')
data.info()  #enflasyon ve işsizlik değişkenlerinin floata dönüştüğünü görmekteyiz
data.tail(2)
data.dropna() 
data.tail()   
data.dropna(how="any") #herhangi bir satırda NaN değeri varsa o satırı sil demek.
                       #Kalıcı hale getirmek için data.dropna(how="any",inplace=True) şeklinde yazmamız gerekir.
data.dropna(how="all")  #Tüm satır nan değerinde ise siler.Bu veride tüm satırın nan olduğu değer yok.
                        #Son iki satırda tarih değişkeninde değer var.
data.dropna(subset=["issizlik"]) #sadece issizlik oranındaki nan valuları siler.

data.fillna(value=1)  #nüfus.fillna(1) bu şekilde de kullanılabilir.

data["enflasyon"].fillna(2).tail()  #sadece seçilen sütundaki nanları doldurur.
                                    #Kalıcı hale getirmek için data["enflasyon"].fillna(2,inplace=True).tail()
data.dropna(thresh=1) #satırda 2 tane nan olduğu için silmedi.
                       #tresh değerini 2 yapsaydım silecekti.
data["yeni_veri"] = data.enflasyon-data.issizlik #yeni sütunda enflasyondan işsizliği çıkardım.
data.head()
data.insert(2,column="yeni_hesaplama",value=data.enflasyon+data.enflasyon) 
#yeni hesaplamada enflasyon ve işsizliği topladım.insert() metodu ile ekleyeceğım yeni yeri yani 2'yi yazdım.
#0 tarih 1 enflasyon ve yazdığım 2 de yeni_hesaplama oldu
data.head()
data.isnull().tail(5) #son 5 değere baktığımızda nan değerleri rahatlıkla görebiliriz.
#sütundaki toplam NaN değerleri verir.Hangi sütunda kaç tane nan değer olduğunu gösterir.
data.isnull().sum()
data.dropna(how="any",inplace=True) #nan değerleri kalıcı hale getirmem gerekiyor.
#yeni bir sütun açarak, oluşturduğum ranki o sütunun içine yazdım.
data["issizlik_sıralama"]=data["issizlik"].rank().astype("int")
data.tail(10)
#ilk önce hangi sütunda olduğunu belirtiyorum,sonrasında isin parametresinin içine hangi değeri istediğimi yazıyorum.
m=data["Tarih"].isin(["2019-02"]) 
#direk olarak m yazıp çağırdığınızda true,false döner.Aşağıdaki şekilde çağırdığınızda istediğiniz filtreleme yapılmış olur.
data[m]
dusukenflasyon=data["enflasyon"].between(2,5)
data[dusukenflasyon]
data["issizlik"].unique()
data.rename({"enflasyon": "yenienflasyon", "issizlik": "yeniissizlik"}, axis="columns", inplace=True)
data.head()