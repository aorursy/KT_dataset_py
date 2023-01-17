import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt;
tablet=pd.read_csv("../input/finaldataset/tablet.csv")

df = tablet.copy()
df.head()
df.shape
df.info()
df.describe().T
df.isnull().sum()     # RAM'de 12 OnKamera'da 5 veri eksik , 2 değişken de sayısal değişken...
df.Renk.unique()
df.FiyatAraligi.unique()
df.Bluetooth.unique() # Var ve Yok obje tipinde değişken değerlerini regresyonda sayısal veriler ile olasılık hesaplayacağımız için 

                      #get_dummies ile 0 1 haline dönüştüreceğiz.
df.CiftHat.unique()
df["4G"].unique()
df["3G"].unique()
df.WiFi.unique()
df.Dokunmatik.unique()
df.FiyatAraligi.value_counts()
df.hist(); 

#OnkameraMP dağılımı normal değil gibi gözüküyor.Onun dışında mikroislemci hızının bir değerinde ciddi bi yoğunluk var.

#ÇekirdekSayısı değişkeninde bazı değerlerden hiç bulunmuyor.
y_order = ["Çok Ucuz","Ucuz","Normal","Pahalı"]
sns.boxplot(x="FiyatAraligi",y="BataryaOmru",data=df,order = y_order,orient="best");
sns.boxplot(x="FiyatAraligi",y="BataryaOmru",data=df,hue="Renk",order = y_order,orient="best");

#Farklı renkteki ucuz ürünlerin BataryaOmru renklere göre değişkenlik göstermiş.
sns.boxplot(x="FiyatAraligi",y="BataryaOmru",data=df,hue="Bluetooth",order = y_order,orient="best");

#Fiyat Araligi Normal olan ürünlere odaklanalım.Bluetooth olan ürünlerin bataryaomru,olmayan ürünlere göre daha yüksek.
sns.boxplot(x="FiyatAraligi",y="BataryaOmru",data=df,hue="CiftHat",order = y_order,orient="best");

#Fiyat Araligi Ucuz olan ürünlere odaklanalım.CiftHat olan ürünlerin bataryaomru,olmayan ürünlere göre daha düşük.
sns.boxplot(x="FiyatAraligi",y="BataryaOmru",data=df,hue="WiFi",order = y_order,orient="best");

#Fiyat Araligi Çok Ucuz olan ürünlere odaklanalım.WiFi olan ürünlerin bataryaomru,olmayan ürünlere göre daha düşük.
sns.boxplot(x="FiyatAraligi",y="BataryaOmru",data=df,hue="Dokunmatik",order = y_order,orient="best");

#Fiyat Araligi Çok Ucuz olan ürünlere odaklanalım.Dokunmatik olan ürünlerin bataryaomru,olmayan ürünlere göre daha düşük.
sns.boxplot(x="FiyatAraligi",y="BataryaOmru",data=df,hue="3G",order = y_order,orient="best");

#FiyatAraligi Pahalı olan ürünler için BataryaOmru , 3G olan ürünlerde , olmayan ürünlere göre daha yüksek.
sns.boxplot(x="FiyatAraligi",y="BataryaOmru",data=df,hue="4G",order = y_order,orient="best");

#Fiyat Araligi Çok Ucuz olan ürünlere odaklanalım.4G olan ürünlerin bataryaomru,olmayan ürünlere göre daha düşük.
sns.boxplot(x="FiyatAraligi",y="CekirdekSayisi",data=df,order=y_order,orient="best");
sns.boxplot(x="FiyatAraligi",y="CekirdekSayisi",data=df,hue="Renk",order=y_order,orient="best");

#Fiyat aralığı çok ucuz olan değişkenlere odaklanalım.Farklı renklerde çekirdek sayısı değeri değişkenlik gösteriyor.
sns.boxplot(x="FiyatAraligi",y="CekirdekSayisi",data=df,hue="Bluetooth",order=y_order,orient="best");

#FiyatAraligi Pahalı olan ürünler için,CekirdekSayisi Bluetooth olan ürünlerde daha düşük.
sns.boxplot(x="FiyatAraligi",y="CekirdekSayisi",data=df,hue="CiftHat",order=y_order,orient="best");

#FiyatAraligi Pahalı olan ürünler için,CekirdekSayisi CiftHat olan ürünlerde daha düşük.
sns.boxplot(x="FiyatAraligi",y="CekirdekSayisi",data=df,hue="Dokunmatik",order=y_order,orient="best");

#FiyatAraligi Çok Ucuz olan ürünler için CekirdekSayisi, Dokunmatik olan ürünlerde daha yüksek.
sns.boxplot(x="FiyatAraligi",y="CekirdekSayisi",data=df,hue="3G",order=y_order,orient="best");

#FiyatAraligi Çok Ucuz olan ürünler için CekirdekSayisi, 3G olan ürünlerde daha düşük.
sns.boxplot(x="FiyatAraligi",y="CekirdekSayisi",data=df,hue="4G",order=y_order,orient="best");

#FiyatAraligi Çok Ucuz olan ürünler için CekirdekSayisi, 4G olan ürünlerde daha düşük.
sns.boxplot(x="FiyatAraligi",y="DahiliBellek",data=df,order=y_order,orient="best");
sns.boxplot(x="FiyatAraligi",y="DahiliBellek",data=df,hue="Renk",order=y_order,orient="best");

#Farklı renklerde olan farklı fiyatlardaki dahilibellek sayıları değişkenlik gösteriyor.
sns.boxplot(x="FiyatAraligi",y="DahiliBellek",data=df,hue="Bluetooth",order=y_order,orient="best");

#FiyatAraligi Çok Ucuz olan ürünler için DahiliBellek, Bluetooth olan ürünlerde daha yüksek.
sns.boxplot(x="FiyatAraligi",y="DahiliBellek",data=df,hue="Dokunmatik",order=y_order,orient="best");

#FiyatAraligi Pahalı olan ürünler için DahiliBellek, Dokunmatik olan ürünlerde daha yüksek.
sns.boxplot(x="FiyatAraligi",y="DahiliBellek",data=df,hue="WiFi",order=y_order,orient="best");

#FiyatAraligi Çok Ucuz olan ürünler için DahiliBellek, WiFi olan ürünlerde daha düşük.(UFak bir fark var)
sns.boxplot(x="FiyatAraligi",y="DahiliBellek",data=df,hue="3G",order=y_order,orient="best");

#FiyatAraligi Pahalı olan ürünler için,DahiliBellek 3G olan ürünlerde daha yüksek.(Çok az fark var.)
sns.boxplot(x="FiyatAraligi",y="DahiliBellek",data=df,hue="4G",order=y_order,orient="best");

#FiyatAraligi Pahalı olan ürünler için,DahiliBellek 4G olan ürünlerde daha yüksek.(Çok az fark var.)
sns.boxplot(x="FiyatAraligi",y="MikroislemciHizi",data=df,order=y_order,orient="best");
sns.boxplot(x="FiyatAraligi",y="MikroislemciHizi",data=df,order=y_order,hue="Renk",orient="best");

#Farklı renklerde ve farklı fiyat aralığındaki değişkenler , MikroislemciHizina göre değişkenlik gösteriyor.
sns.boxplot(x="FiyatAraligi",y="MikroislemciHizi",data=df,order=y_order,hue="Bluetooth",orient="best");

#FiyatAraligi Pahalı olan ürünler için,MikroislemciHizi Bluetooth olan ürünlerde daha yüksek.
sns.boxplot(x="FiyatAraligi",y="MikroislemciHizi",data=df,order=y_order,hue="Dokunmatik",orient="best");

#FiyatAraligi Çok Ucuz olan ürünler için MikroislemciHizi, Dokunmatik olan ürünlerde daha yüksek.
sns.boxplot(x="FiyatAraligi",y="MikroislemciHizi",data=df,order=y_order,hue="WiFi",orient="best");

#FiyatAraligi Çok Ucuz olan ürünler için MikroislemciHizi, WiFi olan ürünlerde daha düşük.
sns.boxplot(x="FiyatAraligi",y="MikroislemciHizi",data=df,order=y_order,hue="3G",orient="best");

#FiyatAraligi Pahalı olan ürünler için,MikroislemciHizi 3G olan ürünlerde daha yüksek.
sns.boxplot(x="FiyatAraligi",y="MikroislemciHizi",data=df,order=y_order,hue="4G",orient="best");

#FiyatAraligi Pahalı olan ürünler için MikroislemciHizi, 4G olan ürünlerde daha düşük.
sns.boxplot(x="FiyatAraligi",y="Agirlik",data=df,order=y_order,orient="best");
sns.boxplot(x="FiyatAraligi",y="Agirlik",data=df,hue="Renk",order=y_order,orient="best");

#Farklı renklerde ve farklı fiyat aralıklarında ürünler için,Ağırlık değişkenlik gösteriyor.
sns.boxplot(x="FiyatAraligi",y="Agirlik",data=df,hue="Bluetooth",order=y_order,orient="best");

#Fiyat Aralığı Pahalı olan ürünler için Agirlik,Bluetooth olan ürünlerde daha düşük.
sns.boxplot(x="FiyatAraligi",y="Agirlik",data=df,hue="Dokunmatik",order=y_order,orient="best");

#Fiyat Aralığı Pahalı olan ürünler için Agirlik,Dokunmatik olan ürünlerde daha yüksek.(Az bir farkla)
sns.boxplot(x="FiyatAraligi",y="Agirlik",data=df,hue="WiFi",order=y_order,orient="best");

#Fiyat Aralığı Pahalı olan ürünler için Agirlik,WiFi olan ürünlerde daha yüksek.
sns.boxplot(x="FiyatAraligi",y="Agirlik",data=df,hue="3G",order=y_order,orient="best");

#Fiyat Aralığı Pahalı olan ürünler için Agirlik,3G olan ürünlerde daha düşük.
sns.boxplot(x="FiyatAraligi",y="Agirlik",data=df,hue="4G",order=y_order,orient="best");

#Fiyat Aralığı Pahalı olan ürünler için Agirlik,4G olan ürünlerde daha yüksek.
sns.boxplot(x="FiyatAraligi",y="Kalinlik",data=df,order=y_order,orient="best");
sns.boxplot(x="FiyatAraligi",y="Kalinlik",data=df,hue="Renk",order=y_order,orient="best");

#Farklı renk ve farklı FiyatAraligindaki ürünlerde,Kalinlik değişkenlik gösteriyor.
sns.boxplot(x="FiyatAraligi",y="Kalinlik",data=df,hue="Bluetooth",order=y_order,orient="best");

#Fiyat Aralığı Çok Ucuz olan ürünlerde Kalinlik,Bluetooth olan ürünlerde daha düşük.
sns.boxplot(x="FiyatAraligi",y="Kalinlik",data=df,hue="Dokunmatik",order=y_order,orient="best");

#Fiyat Aralığı Ucuz olan ürünlerde Kalinlik,Dokunmatik olan ürünlerde daha düşük.
sns.boxplot(x="FiyatAraligi",y="Kalinlik",data=df,hue="WiFi",order=y_order,orient="best");

#Fiyat Aralığı Normal olan ürünlerde Kalinlik,WiFi olan ürünlerde daha düşük.
sns.boxplot(x="FiyatAraligi",y="Kalinlik",data=df,hue="3G",order=y_order,orient="best");

#Fiyat Aralığı Ucuz olan ürünlerde Kalinlik,3G olan ürünlerde daha düşük.
sns.boxplot(x="FiyatAraligi",y="Kalinlik",data=df,hue="4G",order=y_order,orient="best");

#Fiyat Aralığı  Ucuz olan ürünlerde Kalinlik,4G olan ürünlerde daha düşük.
sns.boxplot(x="FiyatAraligi",y="CozunurlukYükseklik",data=df,order=y_order,orient="best");
sns.boxplot(x="FiyatAraligi",y="CozunurlukYükseklik",data=df,hue="Renk",order=y_order,orient="best");

#Farklı renklerde ve farklı fiyat araliklarında CozunurlukYükseklik değerleri değişkenlik gösteriyor.
sns.boxplot(x="FiyatAraligi",y="CozunurlukYükseklik",data=df,hue="Bluetooth",order=y_order,orient="best");

#FiyatAraligi Normal olan ürünlerde CozunurlukYükseklik,Bluetooth olan ürünlerde daha yüksek.
sns.boxplot(x="FiyatAraligi",y="CozunurlukYükseklik",data=df,hue="Dokunmatik",order=y_order,orient="best");

#FiyatAraligi Ucuz olan ürünlerde CozunurlukYükseklik,Dokunmatik olan ürünlerde daha yüksek.
sns.boxplot(x="FiyatAraligi",y="CozunurlukYükseklik",data=df,hue="WiFi",order=y_order,orient="best");

#FiyatAraligi Pahalı olan ürünlerde CozunurlukYükseklik,WiFi olan ürünlerde daha yüksek.
sns.boxplot(x="FiyatAraligi",y="CozunurlukYükseklik",data=df,hue="3G",order=y_order,orient="best");

#FiyatAraligi Çok Ucuz olan ürünlerde CozunurlukYükseklik,3G olan ürünlerde daha yüksek.
sns.boxplot(x="FiyatAraligi",y="CozunurlukYükseklik",data=df,hue="4G",order=y_order,orient="best");

#FiyatAraligi Ucuz olan ürünlerde CozunurlukYükseklik,4G olan ürünlerde daha düşük.
sns.boxplot(x="FiyatAraligi",y="CozunurlukGenislik",data=df,order=y_order,orient="best");
sns.boxplot(x="FiyatAraligi",y="CozunurlukGenislik",data=df,hue="Renk",order=y_order,orient="best");

#Farklı renklerde ve farklı fiyat aralıklarındaki ürünlerde CozunurlukGenislik değerleri değişkenlik gösteriyor.
sns.boxplot(x="FiyatAraligi",y="CozunurlukGenislik",data=df,hue="Bluetooth",order=y_order,orient="best");

#FiyatAraligi Ucuz olan ürünler için CozunurlukGenislik,Bluetooth olan ürünlerde daha düşük.
sns.boxplot(x="FiyatAraligi",y="CozunurlukGenislik",data=df,hue="Dokunmatik",order=y_order,orient="best");

#FiyatAraligi Çok Ucuz olan ürünler için CozunurlukGenislik,Dokunmatik olan ürünlerde daha düşük.(Fark oldukça az)
sns.boxplot(x="FiyatAraligi",y="CozunurlukGenislik",data=df,hue="WiFi",order=y_order,orient="best");

#FiyatAraligi Normal olan ürünler için CozunurlukGenislik,WiFi olan ürünlerde daha yüksek.
sns.boxplot(x="FiyatAraligi",y="CozunurlukGenislik",data=df,hue="3G",order=y_order,orient="best");

#FiyatAraligi Çok Ucuz olan ürünler için CozunurlukGenislik,3G olan ürünlerde daha düşük.
sns.boxplot(x="FiyatAraligi",y="CozunurlukGenislik",data=df,hue="4G",order=y_order,orient="best");

#FiyatAraligi Çok olan ürünler için CozunurlukGenislik,4G olan ürünlerde daha düşük.

#Çok az bir fark var ve diğer fiyat aralıklarında da neredeyse eşit.4G olup olmaması CozunurlukGenislik değerlerinde anlamlı bir farklılık oluşturmuyor.
sns.boxplot(x="FiyatAraligi",y="BataryaGucu",data=df,order=y_order,orient="best");
sns.boxplot(x="FiyatAraligi",y="BataryaGucu",data=df,hue="Renk",order=y_order,orient="best");

#Farklı renklerde ve farklı fiyat aralıklarında BataryaGucu,değişkenlik gösteriyor.
sns.boxplot(x="FiyatAraligi",y="BataryaGucu",data=df,hue="Bluetooth",order=y_order,orient="best");

#FiyatAraligi Ucuz olan ürünlerde BataryaGucu,Bluetooth olan ürünlerde daha düşük.
sns.boxplot(x="FiyatAraligi",y="BataryaGucu",data=df,hue="Dokunmatik",order=y_order,orient="best");

#FiyatAraligi Pahalı olan ürünlerde BataryaGucu,Dokunmatik olan ürünlerde daha düşük.
sns.boxplot(x="FiyatAraligi",y="BataryaGucu",data=df,hue="WiFi",order=y_order,orient="best");

#FiyatAraligi Ucuz olan ürünlerde BataryaGucu, WiFi olan ürünlerde daha yüksek.
sns.boxplot(x="FiyatAraligi",y="BataryaGucu",data=df,hue="3G",order=y_order,orient="best");

#FiyatAraligi Ucuz olan ürünlerde BataryaGucu,3G olan ürünlerde daha yüksek.
sns.boxplot(x="FiyatAraligi",y="BataryaGucu",data=df,hue="4G",order=y_order,orient="best");

#FiyatAraligi Normal olan ürünlerde BataryaGucu,4G olan ürünlerde daha yüksek.
sns.boxplot(x="FiyatAraligi",y="ArkaKameraMP",data=df,order=y_order,orient="best");
sns.boxplot(x="FiyatAraligi",y="ArkaKameraMP",data=df,hue="Renk",order=y_order,orient="best");

#Farklı renklerde ve farklı fiyataraliklarında ArkaKameraMP,değişkenlik gösteriyor.
sns.boxplot(x="FiyatAraligi",y="ArkaKameraMP",data=df,hue="Bluetooth",order=y_order,orient="best");

#FiyatAraligi Ucuz olan ürünlerde ArkaKameraMP,Bluetooth olan ürünlerde daha yüksek.
sns.boxplot(x="FiyatAraligi",y="ArkaKameraMP",data=df,hue="Dokunmatik",order=y_order,orient="best");

#FiyatAraligi Çok Ucuz olan ürünlerde ArkaKameraMP,Dokunmatik olan ürünlerde daha düşük.
sns.boxplot(x="FiyatAraligi",y="ArkaKameraMP",data=df,hue="WiFi",order=y_order,orient="best");

#FiyatAraligi Çok Ucuz olan ürünlerde ArkaKameraMP,WiFi olan ürünlerde daha düşük.(Az fark,ama MP ortalamasına göre yeterli bir fark oluşturuyor.)
sns.boxplot(x="FiyatAraligi",y="ArkaKameraMP",data=df,hue="3G",order=y_order,orient="best");

#FiyatAraligi Çok Ucuz olan ürünlerde ArkaKameraMP,3G olan ürünlerde daha yüksek.
sns.boxplot(x="FiyatAraligi",y="ArkaKameraMP",data=df,hue="4G",order=y_order,orient="best");

#FiyatAraligi Pahalı olan ürünlerde ArkaKameraMP,4G olan ürünlerde daha düşük.
sns.boxplot(x="FiyatAraligi",y="OnKameraMP",data=df,order=y_order,orient="best");
sns.boxplot(x="FiyatAraligi",y="OnKameraMP",hue="Renk",data=df,order=y_order,orient="best");

#Farklı renklerde ve farklı fiyat araliklarında OnKameraMP,değişkenlik gösteriyor.
sns.boxplot(x="FiyatAraligi",y="OnKameraMP",hue="Bluetooth",data=df,order=y_order,orient="best");

#OnKameraMP Bluetooth olup olmamasına göre değişkenlik göstermiyor.
sns.boxplot(x="FiyatAraligi",y="OnKameraMP",hue="Dokunmatik",data=df,order=y_order,orient="best");

#Fiyat Aralığı normal olan ürünlerde OnKameraMP,ufak değişkenlikler gösteriyor.
sns.boxplot(x="FiyatAraligi",y="OnKameraMP",hue="WiFi",data=df,order=y_order,orient="best");

#OnKameraMP WiFi özelliğinin olup olmamasına göre ufak değişkenlikler gösterse de ortalamayı pek etkilemiyor.
sns.boxplot(x="FiyatAraligi",y="OnKameraMP",hue="3G",data=df,order=y_order,orient="best");

#OnKameraMP 3G özelliğinin olup olmamasına göre ciddi değişkenlikler göstermiyor.
sns.boxplot(x="FiyatAraligi",y="OnKameraMP",hue="4G",data=df,order=y_order,orient="best");

#OnKameraMP 4G özelliğinin olup olmamasına göre ciddi değişkenlikler göstermiyor.
numeric=df.drop(["Bluetooth","CiftHat","4G","3G","Dokunmatik","FiyatAraligi","Renk","WiFi"],axis=1)

numeric = pd.DataFrame(numeric)

sns.pairplot(numeric);

#Daha koyu mavi olan grafikler değişkenin dağılım grafikleridir.(histogram)

#Sol en üstteki ilk grafik BataryaGucu'nün dağılım grafiği,sağa doğru bakıldığında ise diğer değişkenler ile olan dağılım grafiğidir(scatterplot)
sns.jointplot(x="ArkaKameraMP",y="OnKameraMP",data=df);#Jointplot ile iki dağılımı eşleştiriyoruz.
sns.jointplot(x="CozunurlukGenislik",y="CozunurlukYükseklik",data=df);
sns.jointplot(x="Agirlik",y="Kalinlik",data=df);
print("BataryaGucu Hızı Ortalaması :",df.BataryaGucu.mean(),"\nBataryaGucu Standart Sapması :",df.BataryaGucu.std(),"\nBataryaGucu Medyanı :",df.BataryaGucu.median())
print("MikroislemciHizi Ortalaması :",df.MikroislemciHizi.mean(),"\nMikroislemciHizi Standart Sapması :",df.MikroislemciHizi.std(),"\nMikroislemciHizi Medyanı :",df.MikroislemciHizi.median())
print("OnKameraMP Ortalaması :",df.OnKameraMP.mean(),"\nOnKameraMP Standart Sapması :",df.OnKameraMP.std(),"\nOnKameraMP Medyanı :",df.OnKameraMP.median())
print("ArkaKameraMP Ortalaması :",df.ArkaKameraMP.mean(),"\nArkaKameraMP Standart Sapması :",df.ArkaKameraMP.std(),"\nArkaKameraMP Medyanı :",df.ArkaKameraMP.median())
print("DahiliBellek Ortalaması :",df.DahiliBellek.mean(),"\nDahiliBellek Standart Sapması :",df.DahiliBellek.std(),"\nDahiliBellek Medyanı :",df.DahiliBellek.median())
print("Kalınlık Ortalaması :",df.Kalinlik.mean(),"\nKalınlık Standart Sapması :",df.Kalinlik.std(),"\nKalınlık Medyanı :",df.Kalinlik.median())
print("CekirdekSayisi Ortalaması :",df.CekirdekSayisi.mean(),"\nCekirdek Sayisi Sapması :",df.CekirdekSayisi.std(),"\nCekirdekSayisi Medyanı :",df.CekirdekSayisi.median())
print("CozunurlukYukseklik Ortalaması :",df.CozunurlukYükseklik.mean(),"\nCozunurlukYukseklik Sapması :",df.CozunurlukYükseklik.std(),"\nCozunurlukYukseklik Medyanı :",df.CozunurlukYükseklik.median())
print("CozunurlukGenislik Ortalaması :",df.CozunurlukGenislik.mean(),"\nCozunurlukGenislik Sapması :",df.CozunurlukGenislik.std(),"\nCozunurlukGenislik Medyanı :",df.CozunurlukGenislik.median())
print("RAM Ortalaması :",df.RAM.mean(),"\nRAM Standart Sapması :",df.RAM.std(),"\nRAM Medyanı :",df.RAM.median())
print("BataryaOmru Ortalaması :",df.BataryaOmru.mean(),"\nBataryaOmru Standart Sapması :",df.BataryaOmru.std(),"\nBataryaOmruMedyanı :",df.BataryaOmru.median())
df.OnKameraMP.value_counts().sum()
df.RAM.value_counts().sum()
Q1=df.OnKameraMP.quantile(0.25)

Q3=df.OnKameraMP.quantile(0.75)

IQR = Q3-Q1

altesik = Q1 - 1.5*IQR

ustesik = Q3 + 1.5*IQR

onkameraaykiri=df[(df.OnKameraMP<altesik)|(df.OnKameraMP>ustesik)]

#Çeyrekler açıklığı hesabını yapıp alt esik ve üst esik değerlerini hesapladık.Ve aykırı değerleri yakaladık.

print("OnKameraMP degiskeninde {} aykırı değer var.".format(onkameraaykiri["OnKameraMP"].count()))
Q1=df.RAM.quantile(0.25)

Q3=df.RAM.quantile(0.75)

IQR = Q3-Q1

altesik = Q1 - 1.5*IQR

ustesik = Q3 + 1.5*IQR

ramaykiri = df[(df.RAM<altesik)|(df.RAM>ustesik)]

print("RAM degiskeninde {} aykırı değer var.".format(ramaykiri["RAM"].count()))
df.RAM.value_counts().sum()
df.RAM.fillna(value=df.RAM.mean(),inplace=True); 

#RAM değişkeninin medyan ve ortalaması arasında büyük farklar olmadığı için eksik değerleri ortalamaya göre dolduracağım.
df.OnKameraMP.fillna(value=df.OnKameraMP.median(),inplace=True);

#OnKameraMP değişkeninin medyan ve ortalaması arasında %25'lik bir fark var ve gözlem sayısı 1995 olan bir değer için yüksek buldum.

#O yüzden değerleri medyan değeri ile dolduracağım.
df.RAM.isnull().sum()
df.RAM.isnull().sum()
from sklearn.preprocessing import OneHotEncoder
df.FiyatAraligi=pd.Categorical(df.FiyatAraligi,ordered=True,categories = ["Çok Ucuz","Ucuz","Normal","Pahalı"]) 

# Ordinal kategorik değişken olarak değiştirdik. Ve sıraladık.
df.Renk = pd.Categorical(df.Renk,ordered=False)

df.Renk #Nominal,yani değişkenler değerleri arasında hiyerarşik sıralamaya sahip değil.
df.info()
ohe = OneHotEncoder() 

#Renk kategorik değişkenini onehotencoding yöntemi ile binary tabana çevireceğim.LabelEncoding yöntemine göre daha iyi sonuçlar elde ediliyor.
df.Renk = ohe.fit_transform(df[["Renk"]]).toarray()
binary = ["Bluetooth","CiftHat","3G","4G","Dokunmatik","WiFi"]
dft=pd.get_dummies(df[binary])

dft.drop(columns=["Bluetooth_Yok","CiftHat_Yok","3G_Yok","4G_Yok","Dokunmatik_Yok","WiFi_Yok"],inplace=True)
dfnew = pd.concat([df,dft],axis=1)

dfnew.drop(columns=["Bluetooth","CiftHat","3G","4G","Dokunmatik","WiFi"],axis=1,inplace=True)

dfnew
korelasyon=df.corr()

korelasyon
sns.heatmap(korelasyon);

#ArkaKameraMP ve OnKameraMP arasında orta şiddette bir korelasyon kuvveti vardır.
sns.lmplot(x="ArkaKameraMP",y="OnKameraMP",data=df);
from sklearn.model_selection import train_test_split
dfnew
X = dfnew.drop(["FiyatAraligi"],axis=1)

y = dfnew["FiyatAraligi"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=30);
X_train
from sklearn.metrics import classification_report , confusion_matrix as cm

#Classification_report model bilgilerini detaylıca inceler.

#Karmaşıklık matrisi tahmin edilen doğru ve yanlış değerler hakkında bilgi verir.Yorumlamasını her modelde yapacağız.
from sklearn.linear_model import LogisticRegression # Bağımlı değişkenimiz ordinal kategorik bir değişken
logreg = LogisticRegression(solver="liblinear") # Liblinear solve yöntemi küçük datasetlerde modelin daha verimli sonuç almasını sağlıyor.
model = logreg.fit(X_train,y_train);
y_pred = model.predict(X_test)

df_y=pd.DataFrame([y_pred,y_test],index=["y_pred","y_test"])

df_y

#Tahmin ettiğimiz değerlerle gerçek y değelerinin bir kısmına bakalım.
model.score(X_test,y_test) #Accuracy Score 
model.coef_ # Bağımsız değişkenlerin katsayıları = b1 değerleri
model.intercept_ #Bağımlı değişkenin katsayısı = b0 katsayısı
print(classification_report(y_test,y_pred))

#f1_score Algoritmanın ne kadar doğru sonuç verdiğini 1 üzerinden hesaplar.

#Precision karmaşıklık matrisinde gördüğümüz asıl durum evet olduğunda verdiği doğru tahmin oranıdır.
print(model.predict_proba(X_test)[0:10])

#1.satırdaki bütün sütunlara odaklanalım.4 farklı değer var bu değerler pd.Categorical ile sıralı bir şekilde belirlediğimiz

# Baştan sona --> Çok ucuz , Ucuz , Normal , Pahalı değişken değerlerinin, tahmin edilme olasılığı.

#y_test te ilk değişken değeri Pahalıydı ve doğru tahmin edildi.Tahmin edilme olasılığı --> 6.72497624e-07
y_pred = pd.Categorical(y_pred,ordered=True,categories=["Çok Ucuz","Ucuz","Normal","Pahalı"])

y_pred[0:10] # y_pred'in değişken tipini kategorik değişken olarak sıraladık.
y_test[0:10]
cmLoj = cm(y_test,y_pred)

cmLoj
pd.DataFrame(cmLoj,index=["Çok Ucuz","Ucuz","Normal","Pahalı"],columns=["Çok Ucuz","Ucuz","Normal","Pahalı"])
from sklearn.naive_bayes import GaussianNB
naivebayes = GaussianNB()
nb_model = naivebayes.fit(X_train,y_train)
nb_pred = nb_model.predict(X_test)
nb_pred;
nb_model.score(X_test,y_test)
nb_cm=cm(y_test,nb_pred)

nb_cm
pd.DataFrame(nb_cm,index=["Çok Ucuz","Ucuz","Normal","Pahalı"],columns=["Çok Ucuz","Ucuz","Normal","Pahalı"])
#Naive-Bayes modeli

# Çok Ucuz değerlerimizi 82 kez Çok Ucuz , 14 kez Ucuz , 29 kez Normal ve 0 kez pahalı olarak tahmin etmiş.

# Ucuz değerlerimizi 9 kez Çok Ucuz , 115 kez Ucuz , 3 kez Normal , 0 kez Pahalı olarak tahmin etmiş.

# Normal değerlerimizi 21 kez Çok Ucuz , 0 kez Ucuz , 98 kez Normal , 6 kez Pahalı olarak tahmin etmiş.

# Pahalı değerlerimiz 0 kez Çok Ucuz , 0 kez Ucuz , 13 kez Normal , 110 kez Pahalı olarak tahmin etmiş.
print(classification_report(y_test,nb_pred))
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(random_state=30)
tree_model = dtree.fit(X_train,y_train)
tree_pred = tree_model.predict(X_test)

tree_pred = pd.Categorical(tree_pred,ordered=True,categories=["Çok Ucuz","Ucuz","Normal","Pahalı"])

tree_pred 

#Düzenli görünmesi açısından ben kategorik değişkene çevireceğim.Tüm model tahminlerini..
X_train.head()     #random_state=30 yazarak x_train'deki değerleri kod her çalıştığında aynı alıyoruz.
tree_model.score(X_test,y_test)
treecm = cm(y_test,tree_pred)

treecm
pd.DataFrame(treecm,columns=["Çok Ucuz","Ucuz","Normal","Pahalı"],index=["Çok Ucuz","Ucuz","Normal","Pahalı"])
# Decision-tree(criterion = "gigi")modeli

# Çok Ucuz değerlerimizi 91 kez Çok Ucuz , 20 kez Ucuz , 14 kez Normal ve 0 kez pahalı olarak tahmin etmiş.

# Ucuz değerlerimizi 9 kez Çok Ucuz , 118 kez Ucuz , 0 kez Normal , 0 kez Pahalı olarak tahmin etmiş.

# Normal değerlerimizi 16 kez Çok Ucuz , 0 kez Ucuz , 98 kez Normal , 11 kez Pahalı olarak tahmin etmiş.

# Pahalı değerlerimiz 0 kez Çok Ucuz , 0 kez Ucuz , 12 kez Normal , 111 kez Pahalı olarak tahmin etmiş.
pd.DataFrame(treecm,index=["Çok Ucuz","Ucuz","Normal","Pahalı"],columns=["Çok Ucuz","Ucuz","Normal","Pahalı"])

#Pahalı değişkenlerimizi , decisionTree(criterion = "gigi") modeli , 11 kez normal 111 kez pahalı olarak tahmin etmiş.
print(classification_report(y_test,tree_pred))#Accuracy,yani algoritmanın doğru sonuç yüzdesi.Precision ve recallın harmonik ortalamasıdır.
dtree2 = DecisionTreeClassifier(criterion="entropy",random_state=30) 

#Parametreyi değiştirdiğimizde accscore arttı.
tree2_model = dtree2.fit(X_train,y_train)
X_train.head()

#Train_test_split fonksiyonunu kullanırken de random_state değerini 30 aldık.Bu çok önemli.

#Çünkü model,eğitim setinde aynı değerleri kullanmaz ise accuracy score dahil bütün tahmin değerleri değişecektir.

#Dolayısıyla yapacağımız kıyaslamanın doğruluğundan bahsedemeyiz.
tree2_pred = tree2_model.predict(X_test)

tree2_pred = pd.Categorical(tree2_pred,ordered=True,categories=["Çok Ucuz","Ucuz","Normal","Pahalı"])

tree2_pred
tree2_model.score(X_test,y_test)
tree2_cm = cm(y_test,tree2_pred)

tree2_cm
pd.DataFrame(tree2_cm,index=["Çok Ucuz","Ucuz","Normal","Pahalı"],columns=["Çok Ucuz","Ucuz","Normal","Pahalı"])
# DecisionTree(criterion = "Entropy") modeli

# Çok Ucuz değerlerimizi 102 kez Çok Ucuz , 9 kez Ucuz , 14 kez Normal ve 0 kez pahalı olarak tahmin etmiş.

# Ucuz değerlerimizi 15 kez Çok Ucuz , 112 kez Ucuz , 0 kez Normal , 0 kez Pahalı olarak tahmin etmiş.

# Normal değerlerimizi 8 kez Çok Ucuz , 0 kez Ucuz , 109 kez Normal , 8 kez Pahalı olarak tahmin etmiş.

# Pahalı değerlerimiz 0 kez Çok Ucuz , 0 kez Ucuz , 11 kez Normal , 112 kez Pahalı olarak tahmin etmiş.
print(classification_report(y_test,tree2_pred)) #Entropy Parametreli decisiontree algoritması modelinin başarı değerlendirmesi
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()#n değeri ön tanımlı olarak 5.
knn_model = knn.fit(X_train,y_train)
knnpred = knn_model.predict(X_test)

knnpred = pd.Categorical(knnpred,ordered=True,categories=["Çok Ucuz","Ucuz","Normal","Pahalı"])

knnpred
knnscore=knn_model.score(X_test,y_test)

knnscore
knncm=cm(y_test,knnpred)

knncm
pd.DataFrame(knncm,index=["Çok Ucuz","Ucuz","Normal","Pahalı"],columns=["Çok Ucuz","Ucuz","Normal","Pahalı"])
#KNN modeli;

# Çok Ucuz değerlerimizi 106 kez Çok Ucuz , 8 kez Ucuz , 11 kez Normal ve 0 kez pahalı olarak tahmin etmiş.

# Ucuz değerlerimizi 12 kez Çok Ucuz , 115 kez Ucuz , 0 kez Normal , 0 kez Pahalı olarak tahmin etmiş.

# Normal değerlerimizi 2 kez Çok Ucuz , 0 kez Ucuz , 122 kez Normal , 1 kez Pahalı olarak tahmin etmiş.

# Pahalı değerlerimiz 0 kez Çok Ucuz , 0 kez Ucuz , 8 kez Normal , 115 kez Pahalı olarak tahmin etmiş.
print(classification_report(y_test,knnpred))
knn_score = []  #Scoreları ekleyeceğimiz listeyi oluşturuyoruz.
i = 2;

if(len(knn_score)<12):

    while(i<15): 

        knn = KNeighborsClassifier(n_neighbors=i)

        knn_model = knn.fit(X_train,y_train)

        knn_pred = knn_model.predict(X_test)

        knn_score.append(knn_model.score(X_test,y_test))

        print("n = {} için".format(i))

        print("Acc Score -->",knn_model.score(X_test,y_test))

        i = i+1

else:

    print("Bu kodu daha önce çalıştırdınız.")

           

            

 #if bloğunu kod tekrardan çalıştırılırsa listeye ekleme yapmaya devam etmesin diye yazdım.

 # KNN fonksiyonunun içindeki n komşu sayısı parametresi 2 den başlayıp 15 e kadar artıp her n değerinde score oluşturuyor.

 # Oluşturduğu scoreları yukarıda oluşturduğumuz diziye ekliyor..

      
knn_score # n'e göre değişen accuracy scoreları listenin içine ekledik ..
plt.plot(knn_score);

plt.title("n değeri değiştikçe AccuracyScore değerinin değişim gözlemi");
basarıdurum = [model.score(X_test,y_test),knnscore,nb_model.score(X_test,y_test),tree2_model.score(X_test,y_test),tree_model.score(X_test,y_test)]

basarıdurum
dfAcc=pd.DataFrame(basarıdurum,index=["LogisticRegression","Knn n=5","NaiveBayes","Tree-entropy","Tree-gini"],columns=["Acc.Score"])

dfAcc
dfAcc.describe().T       

#Başarı durumunu df'ye çevirdikten sonra içlerine her modelde elde ettiğimiz acc.scoreu ekledik. 

#ve df.describe()ile değerlerin istatistiksel yorumunu yaptık.