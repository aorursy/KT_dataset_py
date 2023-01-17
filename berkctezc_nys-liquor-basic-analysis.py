import numpy as np # matris formatında manipülasyonların yapılabilmesi icin

import pandas as pd # csv veriyi okuma, isleme, veriden elde edilen bilgileri kullanarak sonuc cıkarabilmek icin

import matplotlib as plot #veriyi gorsel grafiklerle zenginlestirebilmek icin



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# Veriyi Import Etme

#dataset = pd.read_csv('nys-liquor-authority-new-applications-received.csv') 

#jupyter ile ayni directoryde ise bu sekilde cagirabiliriz

#not: ipython için tam directory gerekebilmektedir

dataset = pd.read_csv('../input/nys-liquor-authority-new-applications-received.csv') #kaggle
dataset.head() # ilk 5 satirin onizlemesi
dataset.tail() # son 5 satirin onizlemesi
dataset.sample(5) #rastgele secilmis 5 satir

                  #birden fazla calistirarak verimiz hakkında genel bir bilgi edinebiliriz
dataset.describe() #cesitli istatistiki bilgiler
dataset.shape #datamızın satır ve sütun sayısı
dataset.info()
dataset.isna().any() # Hangi sütunların null deger tasidigi bilgisi
#Sütun başına kaç null değer var?



nulls = dataset.isna().sum()

nulls[nulls > 0]

#Null degeri olan sütunlar ve kac tane null deger oldugu

#bu null degerleri doldurmamız gerekir yoksa bizi istatistik ve grafik ciziminde yanıltabilirler
data = dataset.fillna(method="ffill", inplace=False,limit=3) 

# inplace false yapıp datanın orjinal haline dokunmamış olduk

# ve data adli dizimiz artik null degerlerin bircogundan arınmıs dataset olarak 

# elimizde bulunmakta





#not:ayrıca Null valueya sahip olan degerleri drop edip istatistige hic katmamak gibi bir secenegimiz de mevcut

#data.dropna(inplace=True)





# fillna fonksiyonunun degisik metodları vardır 

# benim kullandıgım ffill (forward fill) metodu bir sonraki degere bakıp doldurur

# (diger metodlardan daha cok null doldurdugu icin ffill'i sectim)

# limit parametresi ise ardısık olarak kac tane ile sınırlandırılması gerektigidir

# limiti ne kadar düsük verirsek dagilimi yayilacagindan gercege daha yakın sonuc elde ederiz

nulls = data.isna().sum()

nulls[nulls > 0]
data.fillna('unknown',inplace = True)

nulls = data.isna().sum()

nulls[nulls > 0]
data.info()
data.columns

#dataframedeki bütün sütunları verir ve sonradan bunlar arasında karşılaştırma vs. islemler yapilabilir
data.nunique() #  bu fonksiyon ile her sütundaki birbirinden farklı (benzersiz/unique) degerlerin sayisini görüyoruz
data['License Type Name'].value_counts() # verdigimiz sütundaki unique degerler ve tekrar sayilari 
data["License Type Name"].value_counts().plot.pie(shadow=True,figsize=[15,15])  



#bu sekildede istedigimiz plot seklinde grafigi gorebiliriz
data['License Received Date'].value_counts().plot.bar(figsize=[20,10])
data["County Name (Licensee)"].value_counts().plot.area(figsize=[20,5],stacked=False)
data['Agency Zone Office Name'].value_counts().plot.bar(figsize=[20,5])

#Bagli oldugu bölgelerin dagiliminin bar plotunda gösterimi
data["License Type Code"].value_counts().plot.area(figsize=[20,5],stacked=False)
data["License Class Code"].value_counts().plot.bar(figsize=[20,5],orientation='horizontal')
# data.info() fonksiyonumuzu cagirip hangi sütunlarımızın hangi veritiplerine sahip oldugunu gorelim

data.info()
data = data.astype({

    'License Type Name':'category','License Type Code':'category',

    'Agency Zone Office Name':'category','County Name (Licensee)':'category',

    'Premises Name':'category','Doing Business As (DBA)':'category',

    'Actual Address of Premises (Address1)':'category','Additional Address Information (Address2)':'category',

    'City':'category','State':'category','License Certificate Number':'category','License Received Date':'category'

})

#stringden direkt olarak integer dönüşümü yapamayacağımız için önce category nesnesine dönüstürüyoruz



cat_columns=data.select_dtypes(['category']).columns

data[cat_columns]=data[cat_columns].apply(lambda x: x.cat.codes)
data.shape

data.info()
data.sample(10)