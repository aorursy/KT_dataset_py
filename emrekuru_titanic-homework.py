#gerekli kütüphanelerin kurulması

#import komutu ile kütüphaneleri çağırıyoruz

#eğer bir kütüphaneden belirli bir bölümü çağıracaksak 'from... import'

import pandas as pd

from pandas import Series, DataFrame 
#kaggle üzerinden veri setinin çağırılması

#kaggle üzerinde yüklü oldugu için veri setini direkt bu şekilde yükledim

titanic_df=pd.read_csv('../input/titanic/train.csv')
#veri setlerine göz atmak için kullanılan bir komut 

#değişkenleri ve gözlem değerlerine kabaca bir bakış yapmak için ideal

titanic_df.head()
#veri setinin yapısı hakkında bilgiler hangi değişken hangi türde 

#kaç adet boş olmayan gözlem içeriyor gibi bilgiler

titanic_df.info()
#veri üzerinde manipulasyon yapmak için gerekli olan diğer kütüphaneleri de aktif ettik

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#veri setinin cinsiyet olarak kırılımının görselleştirilmesi

sns.factorplot('Sex',data=titanic_df,kind='count')

#gemideki erkek yolcu sayısı bayan yolcu sayısının yaklaşık olarak iki katı kadardır 
#cinsiyetlerin kendi içerisinde sınıflara göre kırılımının görselleştirilmesi de aşagıdaki gibidir

sns.factorplot('Sex',data=titanic_df,hue='Pclass',kind='count')

#erkeklerin çogunlugu üçüncü sınıf yolculuk yaparken 

#bu oran kadınlar arasında nispeten daha normal dağılış göstermiştir
#burada ise yolcu sınıflarının içerisindeki cinsiyet dağılımları incelenmiştir

sns.factorplot('Pclass',data=titanic_df,hue='Sex',kind='count',order=[1,2,3])

#birinci ve ikinci sınıf yolcular daha homojen dağılış gösterirken 

#üçüncü sınıf yolcularda erkekler kadınların yaklaşık iki katı kadardır
#apply fonksiyonun tanımlanması bu fonksiyonu sonraki adımda yolcuları yaşına göre ayırmak için kullanıcaz

def male_female_child(passenger):

    age,sex = passenger

    

    if age < 16:

        return 'child'

    else:

        return sex

#eğer yaş 16 dan küçükse 'child'

#değilse kişinin cinsiyetini yazdır
#az önce yazdıgımız fonksiyonu burada kullanıyoruz 

#eger yolcu 16 yasından kucukse yeni açılan kolona 'child' yazdırıyoruz, değilse cinsiyet değerini yazdirıyoruz

titanic_df['person']=titanic_df[['Age','Sex']].apply(male_female_child,axis=1)
#işe yarayıp yaramadıgına bakalım

titanic_df.head(10)

#en son sütuna bakın ve yaş değerleri ile karşılaştırın 

#PassengerId 8 ve 10 yaş değeri 16'dan küçük oldugu için 

#yeni oluşturulan değişken içerisine 'child' olarak kaydedildi
#yolcu sınıflarının dağılımına şimdi de 'person' sütununa göre bakalım

sns.catplot('Pclass',data=titanic_df,kind='count',hue='person')

#en fazla çocuk yolcu 3. sınıf içerisinde bulunmaktadır
#yolcu yaşlarının dağılımını kabaca görebilmek için histogram üzerinden inceleyelim

titanic_df['Age'].hist(bins=70)

#cocuklar içerisinde bebekler daha cok gibi gözükmekte

#genel olarak yolcu yaşları 20 ile 30 arasında dagılmakta
titanic_df['Age'].mean()

#ortalama yaş 30 olarak çıkmakta
#yolcu sayılarının saydırılması kaç tane erkek kaç tane kadın kaç tane çocuk var görebilmek için

titanic_df['person'].value_counts()

#erkekler kadınların neredeyse iki katı kadar

#toplam sayının %10 luk kısmı ise çocuklardan oluşuyormuş
fig = sns.FacetGrid(titanic_df,hue='Sex',aspect=4)

#cinsiyete göre ayırdık, aynı yerde üstüste iki cinsiyet için de dağılışı çizdik

fig.map(sns.kdeplot,'Age',shade=True)

#bunu yaşa göre çizecek

oldest = titanic_df['Age'].max()

fig.set(xlim=(0,oldest))

#x ekseninin nerede başlayıp nereye kadar gideceği 

fig.add_legend()

#genel olarak erkek yolcuların yaşları daha yüksek gibi gözükmekte

#bir üstteki görselletirmeden farklı olarak 'person' değişkenine göre çizdirdik dağılımlarını

fig = sns.FacetGrid(titanic_df,hue='person',aspect=4)

fig.map(sns.kdeplot,'Age',shade=True)

oldest = titanic_df['Age'].max()

fig.set(xlim=(0,oldest))

fig.add_legend()

#cocukların geneli 2 ile 6 bilemedin 7 yaşları arasında dağılıyor

#cocuklar ayrı olarak gözlemlendiginde ise kadın ile erkeğin dağılımı gittikçe birbirine daha cok benzemiş
#yolcu sınıflarına göre dağılım

fig = sns.FacetGrid(titanic_df,hue='Pclass',aspect=4)

fig.map(sns.kdeplot,'Age',shade=True)

oldest = titanic_df['Age'].max()

fig.set(xlim=(0,oldest))

fig.add_legend()

#görüldügü gibi 1. sınıf gayet normal dagılırken 

#2. ve 3. sınıf 20 ve 30 arasına sıkışmış gibi görünüyor
#deck diye bir veri seti olusturuyoruz bunun için titanic veri setinden Cabin değişkeninini alıp boş olanları düşürüyoruz

deck = titanic_df['Cabin'].dropna()
#şu şekilde bir şey oluyor yeni oluşturdugumuz "deck" veri seti 

#bize bunların sadece ilk harfleri yeterli sınıflandırma yapabilmek içim

#oda numaraları bir değişikliğe sebep olmuyor çünkü

deck.head()
#ilk harfleri alıp saklayacagımız vektoru tanımladık

levels = []

#daha sonra söyle bir döngü yazıyoruz

#level için deck deki gözlemlerin 0. değerlerini al 

#onları git levels vektorunun içine yaz

for level in deck:

    levels.append(level[0])

#yeni bir dataframe oluşturduk onu da az once olusturdugumuz levels dan olusturuyoruz    

cabin_df = DataFrame(levels)

cabin_df.columns = ['Cabin']

#burada ise kabinlerin türlerine göre histogram cizdiriyoruz

sns.catplot('Cabin',data=cabin_df,palette='winter_d',kind='count', 

            order=['A','B','C','D','E','F','T'])
#yolcuların gemiye biniş yerlerine göre kırılımları 

#en fazla yükleme (yolcu binişi) southampton limanından olmuş görünüyor 

sns.catplot('Embarked',data=titanic_df,hue='Pclass',

               order=['C','Q','S'],kind='count')
#ara ara neyi nereden yapacagınızı görebilmek için veri setini böyle çağırıp bakmak çok işe yarıyor

titanic_df.head()

#bir sonraki problem kimler ailesi ile kimler yalnız 

#bunun için 'sibsp' yani kardesi yanında mı 

#'parch' ailesi yanında mı sütunlarını inceleyecegiz

#kategorik değişken 0 yok 1 var demek
#yeni bir değişken oluşturuyoruz 'alone' isimli 

#bu iki sutunun karslık geldiği satır değerlerinin toplamı 

#herhangi bir değerde 0 geliyorsa o adam yalnızları oynuyor demektir

titanic_df['Alone'] = titanic_df.SibSp + titanic_df.Parch
#mesela buradaki 3. gözlemimiz yalnız bir yolcu :)

titanic_df['Alone']
#yeni oluşturdugumuz değişkene değerleri atama işlemi 

#eğer toplam 0 dan büyükse 'with family'

#eger toplam 0 dan küçükse 'alone' olarak dolduruyouz

titanic_df['Alone'].loc[titanic_df['Alone'] > 0] = 'With Family'



titanic_df['Alone'].loc[titanic_df['Alone'] == 0] = 'Alone'
#en güncel halimiz artık şu şekilde oldu

titanic_df.head()

#ailesi yanında mı değil mi sütunumuz da geldi
#Survived kolonundaki kategorik değişkeni isimlendirerek yeni bir sütün oluşturduk

# 0 olanlar no 1 olanlar yes olarak işlendi

titanic_df['Survivor'] = titanic_df.Survived.map({0 : 'no', 1 : 'yes'})
#yaklaşık 350 kişi ailesi ile beraberken 

#550 600 yolcu ise bu yolculuğa tek başına çıkmış gibi görünüyor

sns.catplot('Alone',data=titanic_df,palette='Blues',kind='count')
#hayatta kalanlar ile başaramayanların karşılaştırılması

sns.catplot('Survivor',data=titanic_df,palette='Set1',kind='count')

#bu elim olayda 550 kişiden fazlası hayatını kaybederken 

#yalnızca 350 civarı insan hayatta kalmayı başarabilmiştir
#yolcu sınıfları hayatta kalma ile ilgili bir farklılıga sebep oluyor mu bakalım

sns.factorplot('Pclass','Survived',data=titanic_df,order=[1,2,3])

#birinci sınıf yolcularda hayatta kalma oranı yüzde 60 ların üzerindeyken 

#bu oran 3. sınıf yolcularda yüzde 25 lerde 
#yolcunun türü hayatta kalma oranını nasıl etkiliyor

#çocuklar için en yüksek hayatta kalma oranı 2. sınfta

#1. sınıfta yolculuk yapan çocukların hayatta kalma varyansı çok yüksek

#kadın yolcular için sınıf yükseldikçe hayatta kalma oranı düşüyor 

#çünkü kadın yolcu sayısı en yogun 1. sınıfta idi 

sns.factorplot('Pclass','Survived',hue='person',data=titanic_df,order=[1,2,3])
#yaşa göre hayatta kalma oranları nasıl bir 

#eğilim gösteriyor ona da bakalım

#genel olarak düşüş eğiliminde 

#yaş ilerledikçe hayatta kalma oranı daha da düşüyor

sns.lmplot('Age','Survived',data=titanic_df)
#yolcu sınıflarının kendi içlerinde yaşa göre eğilimleri de

#hepsi için yaş ilerledikçe hayatta kalma oranının düştüğü 

#yorumu yapılabilir

sns.lmplot('Age','Survived',hue='Pclass',

           data=titanic_df,palette='winter')
#aralıklar belirledik 

#o aralıklar için de standart sapmalarını gözlemlemek için

#boxplot çizdirdik 

#1 sınıfta ilerleyen yaşlarda hayatta kalma oranı çok değişkenlik göstermekte

generations = [10,20,40,60,80]



sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df,

           palette='winter',x_bins=generations)
#burada enteresan olarak

#cinsiyet kırılımında yaşın hayatta kalma oranı üzerindeki etkisine bakıldıgında

#kadın yolcular için yaş ilerledikçe hayatta kalma oranı yükselmekte

#genel eğilim ise düşme yönündeydi

sns.lmplot('Age','Survived',hue='Sex',data=titanic_df,

           palette='winter',x_bins=generations)
#buradaki grafikte de yalnız insanların 

#hayatta kalma oranları 

#ailesi ile birlikte gelenlerin neredeyse yarısı oranında

sns.factorplot('Alone','Survived',data=titanic_df)