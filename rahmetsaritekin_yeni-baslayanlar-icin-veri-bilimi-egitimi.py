# Bu python 3 ortamı bir çok yardımcı analiz kütüphaneleri ile birlikte kurulu olarak gelmektedir.
# Docker image'i olarak kaggle/python'dadır.(https://github.com/kaggle/docker-python)
# Örneğin, yüklenecek birkaç yardımcı paket var


import numpy as np # cebir
import pandas as pd # veri işleme, CSV dosyaları I/O (örn. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # görselleştirme aracı

# Giriş veri dosyaları "../input/" dizinindedir.
# Örneğin bu hücreyi çalıştırmak için Shift+Enter'a aynı anda basarsanız hücre çalışır ve o dizindeki dosyaları sıralar.
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Geçerli dizine yazdığınız herhangi bir sonuç çıktı olarak kaydedilir.
veri = pd.read_csv('../input/pokemon.csv')
veri.info()
veri.corr()
# korelasyon haritası 
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(veri.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
veri.head(10)
veri.columns
# Çizgi Grafiği
# color = renk, label = etiket, linewidth = çizgi genişliği, alpha = opaklık, grid = ızgara, linestyle = çizgi stili
veri.Speed.plot(kind = 'line', color = 'g',label = 'Speed',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
veri.Defense.plot(color = 'r',label = 'Defense',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = etiketi grafiğe koyar
plt.xlabel('x ekseni')              # label = etiket adı
plt.ylabel('y ekseni')
plt.title('Grafiğin Başlığı')            # title = grafiğin başlığı
plt.show()
# Dağılım Grafiği 
# x = attack, y = defense
veri.plot(kind='scatter', x='Attack', y='Defense',alpha = 0.5,color = 'red')
plt.xlabel('Attack')              # label = etiketin adı
plt.ylabel('Defence')
plt.title('Attack Defense Scatter Plot')            # title = grafiğin başlığı
# Histogram
# bins = şekildeki çubuk sayısı
veri.Speed.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
# clf() = tekrar temizler, temiz bir başlangıç yapabilirsin.
veri.Speed.plot(kind = 'hist',bins = 50)
plt.clf()
# clf() nedeniyle grafiği göremiyoruz.
# sözlüğü oluşturun ve onun anahtarlarına ve değerlerine bakın
sözlük = {'ispanya' : 'madrid','usa' : 'vegas'}
print(sözlük.keys())
print(sözlük.values())
## Anahtarlar, string, boolean, float, integer veya tubles gibi değişmez nesneler olmalı!!
# Liste sabittir.
# Anahtarlar biricik, eşsiz olmalı :)
sözlük['ispanya'] = "barcelona"    # varolan kaydı güncelleme
print(sözlük)
sözlük['fransa'] = "paris"       # yeni kayıt ekleme
print(sözlük)
del sözlük['ispanya']              # 'ispanya' adındaki kaydı silme
print(sözlük)
print('fransa' in sözlük)        # içerir mi kontrol etme
sözlük.clear()                   # sözlükteki tüm verileri silme
print(sözlük)

# Tüm kodu çalıştırmak için alt satırı yorumlamamız gerekiyor.

# del sözlük         # tüm sözlüğü sil
print(sözlük)       # hata verir çünkü sözlüğü yukardaki kodla sildik.
veri = pd.read_csv('../input/pokemon.csv')

seriler = veri['Defense']        # veri['Defense'] = seridir.
print(type(seriler))
veri_cercevesi = veri[['Defense']]  # veri[['Defense']] = veri çerçevesidir.
print(type(veri_cercevesi))

# Karşılaştırma operatörleri
print(3 > 2)
print(3!=2)
# Mantık operatörleri
print(True and False)
print(True or False)
# 1 - Pandas'da veri çevçevelerini filtreleme
x = veri['Defense']>200     # 200'den daha yüksek savunma değerine sahip sadece 3 pokemon var
veri[x]
# 2 - logical_and ile filtreleme
# 200'dan daha yüksek savunma değerine ve 100'den daha yüksek saldırı değerine sahip olan sadece 2 pokemon var.
veri[np.logical_and(veri['Defense']>200, veri['Attack']>100 )]
# Bu, önceki kod satırında da aynıdır. Bu nedenle filtreleme için '&' kullanabiliriz.
veri[(veri['Defense']>200) & (veri['Attack']>100)]
# Koşul (i eşit değil 5) doğruysa döngü devam eder.
i = 0
while i != 5 :
    print('i : ',i)
    i +=1 
print(i,' eşittir 5\'e')
# Koşul (i eşit değil 5) doğruysa döngüde kalır.
lis = [1,2,3,4,5]
for i in lis:
    print('i is: ',i)
print('')

# Listeyi ve liste değerini numaralandır.
# index : değer = 0:1, 1:2, 2:3, 3:4, 4:5 (0'dan başlar.)
for index, deger in enumerate(lis):
    print(index," : ",deger)
print('')   

# Sözlük için ise
# Anahtarın ve sözlüğün değerini elde etmek için döngü için kullanabiliriz. Sözlük kısmında anahtar ve değeri öğrendik.
sozluk = {'ispanya':'madrid','fransa':'paris'}
for anahtar,deger in sozluk.items():
    print(anahtar," : ",deger)
print('')

# Pandas'da index ve değer elde edebiliriz
for index,deger in veri[['Attack']][0:1].iterrows():
    print(index," : ",deger)


# example of what we learn above
def tuble_ex():
    """tanımlı t tuble döndürüldü"""
    t = (1,2,3)
    return t
a,b,c = tuble_ex()
print(a,b,c)
# tahmin edelim bakalım ne çıkıcak _??
x = 2
def f():
    x = 3
    return x
print(x)      # x = 2 global scope
print(f())    # x = 3 local scope
# Peki ya local scope yoksa _??
x = 5
def f():
    y = 2*x        # local bi x yok
    return y
print(f())         # bu yüzden global scope x kullanılır.
# Önce local 'scope' arandı, sonra 'global scope' araştırıldı, eğer hala yoksa son olarak 'built in scope' aranır.
# Peki nasıl built in scope'ları öğrenebiliriz _?
import builtins
dir(builtins)
#iç içe geçmiş fonksiyonlar
def kare():
    """ karenin değerleri döndürülür """
    def ekle():
        """ iki local değişken ekle """
        x = 2
        y = 3
        z = x + y
        return z
    return ekle()**2
print(kare())    
# varsayılan parametreler
def f(a, b = 1, c = 2):
    y = a + b + c
    return y
print(f(5))
# peki varsayılan argümanları değiştirmek istiyorsak
print(f(5,4,3))
# esnek parametreler *args
def f(*args):
    for i in args:
        print(i)
f(1)
print("")
f(1,2,3,4)
# esnek parametreler **kwargs bir sözlüktür
def f(**kwargs):
    """ sözlüğün anahtar ve degerini yazdıralım."""
    for anahtar, deger in kwargs.items():               # Eğer bu bölümü anlamadıysanız döngüler kısmındaki sözlükte döngülere göz atabilirsiniz.
        print(anahtar, " ", deger)
f(ulke = 'ispanya', baskent = 'madrid', nufus = 123456)
# lambda fonksiyonu
kare = lambda x: x**2     # x parametredir
print(kare(4))
toplam = lambda x,y,z: x+y+z   # x,y,z fonsiyonun parametreleridir
print(toplam(1,2,3))
numara_listesi = [1,2,3]
y = map(lambda x:x**2,numara_listesi)
print(list(y))
# yineleme örneği
name = "ronaldo"
it = iter(name)
print(next(it))    # print next iteration
print(*it)         # print remaining iteration

# zip örnekleri
liste1 = [1,2,3,4]
liste2 = [5,6,7,8]
z = zip(liste1,liste2)
print(z)
z_listesi = list(z)
print(z_listesi)
un_zip = zip(*z_listesi)
un_liste1,un_liste2 = list(un_zip) # unzip tuble döndürür
print(un_liste1)
print(un_liste2)
print(type(un_liste2))
# list comprehension örneği
num1 = [1,2,3]
num2 = [i + 1 for i in num1 ]
print(num2)
# yinelenebilir'deki şartlar(if, else)
num1 = [5,10,15]
num2 = [i**2 if i == 10 else i-5 if i < 7 else i+5 for i in num1]
print(num2)
# pokemon csv'ye dönelim ve bir tane daha list comprehension örneği yapalım
# lets classify pokemons whether they have high or low speed. Our threshold is average speed.
# pokemonları yüksek veya düşük hızlara sahip olup olmadığını sınıflayalım. Eşiğimiz ortalama hızdır.
esik = sum(veri.Speed)/len(veri.Speed)
veri["hız_seviyesi"] = ["high" if i > esik else "low" for i in veri.Speed]
veri.loc[:10,["hız_seviyesi","Speed"]] # loc'u daha sonra daha detaylı göreceğiz.
veri = pd.read_csv('../input/pokemon.csv')
veri.head()  # head ilk 5 satırı gösterir
# tail son 5 satırı gösterir
veri.tail()
# columns sütun isimlerini verir
veri.columns
# shape sütun ve satırların sayısını verir
veri.shape
# info veri tipini, sütun ve satır sayısını, hafıza kullanımı bilgilerini verir
veri.info()
# Örneğin pokemom tiplerinin sıklığına bakalım
print(veri['Type 1'].value_counts(dropna =False))  # eğer NaN(boş) değeler varsa onlarda sayılır
# Aşağıda görülebileceği gibi 112 su pokemon veya 70 çim pokemon vardır
# Örneğin, maksimum HP 255 ve minimum savunma 5'tir
veri.describe() # null girdileri görmezden gelir
# Örneğin: efsanevi olan ya da olmayan pokemonların saldırısını karşılaştır
# Üstte siyah çizgi max
# Üstte mavi çizgi %75
# Kırmızı çizgi ortancadır (%50)
# Altta mavi çizgi% 25
# Altta siyah çizgi min
# Ayırıcı yok
veri.boxplot(column='Attack',by = 'Legendary')
# Öncelikle, melt() methodunu daha kolay açıklamak için pokemons verilerinden yeni veriler oluşturuyorum.
yeni_veri = veri.head()    # Yeni veriler için adece 5 satırını aldım 
yeni_veri
# hadi melt'leyelim
# id_vars = melt'lemek isteMEdiğimiz şeyler
# value_vars = melt'lemek istediğimiz şeyler
meltlenmis = pd.melt(frame=yeni_veri,id_vars = 'Name', value_vars= ['Attack','Defense'])
meltlenmis
# Index is name
# Bu sütunların değişken olması istiyorum
# Eveet geri eski haline döndürdük =)
meltlenmis.pivot(index = 'Name', columns = 'variable',values='value')
# Öncelikle 2 ayrı veri seti oluşturalım
veri1 = veri.head()
veri2= veri.tail()
conc_data_row = pd.concat([veri1,veri2],axis =0,ignore_index =True) # axis = 0 : veri setlerini satır olarak altına ekler.
conc_data_row
veri1 = veri['Attack'].head()
veri2= veri['Defense'].head()
conc_data_col = pd.concat([veri1,veri2],axis =1) # axis = 0 : veri setlerini sütun olarak ekler
conc_data_col
veri.dtypes
# lets convert object(str) to categorical and int to float.
veri['Type 1'] = veri['Type 1'].astype('category')
veri['Speed'] = veri['Speed'].astype('float')
# Gördüğünüz gibi, Type 1 object'den categorical'a dönüştürüldü.
# ve Speed int'den float'a döndü.
veri.dtypes
# Hadi bakalım pokemon veri setimizde eksikler var mı :O
# Eveet gördüğünüz burada üzere 800 girdi var. Ancak Type 2'nin 414 tane boş olmayan (non-null) nesnesi var yani 386 tanesi eksik.
veri.info()
# Type 2'ye bakalım
veri["Type 2"].value_counts(dropna =False)
# Gördüğünüz üzere 386 NaN(Eksik) değerimiz var.
# Nan değerleri olduğu gibi bırakalım
veri1 = veri   # ayrıca veriyi, eksik değerleri doldurmak için kullanacağız, bu yüzden veri'yi veri1 değişkenine atadım
veri1["Type 2"].dropna(inplace = True)  # inplace = True onu yeni değişkene atamadığımız anlamına gelir.Değişiklikler otomatik olarak veriye atanır
# Eee çalıştı mı _?
#  Lets check with assert statement
# Hadi bide assert ifadesine bakalım.
# Assert ifadesi:
assert 1==1 # hiçbirşey dönmez çünkü ifade True 
# In order to run all code, we need to make this line comment
# Tüm kodu çalıştırırken aşağıdaki satırı yorum satırı yapamamız gerekmektedir.
# assert 1==2 # hata döndürür çünkü ifade False
assert  veri['Type 2'].notnull().all() # hiçbişey döndürmez çünkü NaN değerleri çıkardık.
veri["Type 2"].fillna('empty',inplace = True)
assert  veri['Type 2'].notnull().all() # yine hiçbişey dönmez çünkü NaN değerleri doldurduk
# With assert statement we can check a lot of thing. For example
# Assert ifadesi ile bir çok şeyi kontrol edebiliriz. Mesela:
# assert data.columns[1] == 'Name'
# assert data.Speed.dtypes == np.int
# sözlüklerden veri seti
sehir = ["İspanya","Fransa"]
nufus = ["11","12"]
list_label = ["sehir","nufus"]
list_col = [sehir,nufus]
zipped = list(zip(list_label,list_col))
data_dict = dict(zipped)
df = pd.DataFrame(data_dict)
df
# Add new columns
df["baskent"] = ["madrid","paris"]
df
# Broadcasting
df["income"] = 0 #Broadcasting tüm sütun
df
# Tüm veriyi çizelim
veri1 = veri.loc[:,["Attack","Defense","Speed"]]
veri1.plot()
# hmm kafa karıştırıcı .s.s
# altplan
veri1.plot(subplots = True)
plt.show()
# dağılım grafiği  
veri1.plot(kind = "scatter",x="Attack",y = "Defense")
plt.show()
# histogram grafiği
veri1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True)
# kümülatif ve kümülatif olmayan histogram altplanı
fig, axes = plt.subplots(nrows=2,ncols=1)
veri1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[0])
veri1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)
plt.savefig('graph.png')
plt
veri.describe()
zaman_listesi = ["1992-03-08","1992-04-12"]
print(type(zaman_listesi[1])) # Sizin de gördüğünüz gibi tarih bir string nesnesi
# Ancak biz date time'ın nesne olmasını istiyoruz
datetime_nesnesi = pd.to_datetime(zaman_listesi)
print(type(datetime_nesnesi))
# close warning
import warnings
warnings.filterwarnings("ignore")
# Pratik yapmak için pokemon verilerinin başını alıp bir zaman listesi ekleyelim
veri2 = veri.head()
zaman_listesi = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]
datetime_nesnesi = pd.to_datetime(zaman_listesi)
veri2["date"] = datetime_nesnesi
# indeks olarak tarih yapalım
veri2= veri2.set_index("date")
veri2 
# Şimdi tarih indeksimize göre seçebiliriz
print(veri2.loc["1993-03-16"])
print(veri2.loc["1992-03-10":"1993-03-16"])
# Geçen bölümde oluşturduğumuz veri2 setini kullanalım.
veri2.resample("A").mean()
# Hadi month ile yeniden örneklendirelim.
veri2.resample("M").mean()
# Görebildiğiniz gibi bir çok NaN değeri var çünkü veri2 seti tüm ayları içermiyor.
# Gerçek hayatta (veriler gerçek. Biz veri2'yi yaratmamışız sanki) bu problemi interpolate ile çözebiliriz
# İlk değerden interpolate edelim. NaN değerleri dolduralım.
# String ve boolean değerleri doldurmuyor fakat int ve float değerleri interpolate'ın içerdiği farklı yöntemler ile dolduruyor. Örneğin 'linear' olarak
veri2.resample("M").first().interpolate("linear")
# birde mean() methodu ile interpolate edelim.
veri2.resample("M").mean().interpolate("linear")
# veriyi okuyalım
veri = pd.read_csv('../input/pokemon.csv')
veri = veri.set_index("#")
veri.head()
# köşeli parantez kullanarak indeksleme
veri["HP"][1]
# sütun niteliğini ve satır etiketini kullanma
veri.HP[1]
# Loc erişimcisini kullanma
veri.loc[1,["HP"]]
# Sadece belirli sütunları seçme
veri[["HP","Attack"]]
# Sütun seçimi ile arasındaki fark : seriler ve veri setleri
print(type(veri["HP"]))     # seriler
print(type(veri[["HP"]]))   # veri setleri
# Serileri dilimleme ve indeksleme
veri.loc[1:10,"HP":"Defense"]   # 10 and "Defense" are inclusive
# Tersten dilimleme
veri.loc[10:1:-1,"HP":"Defense"] 
# Bir şeyden sonra
veri.loc[1:10,"Speed":] 
# Boolean seriler yaratma
boolean = veri.HP > 200
veri[boolean]
# Filtreleri birleştirme
ilk_filtre = veri.HP > 150
ikinci_filtre = veri.Speed > 35
veri[ilk_filtre & ikinci_filtre]
# Diğerlerine göre sütun filtreleme
veri.HP[veri.Speed<15]
# Normal python fonksiyonları
def div(n):
    return n/2
veri.HP.apply(div)
# Lambda fonksiyonu: her bir öğeye rastgele python işlevi uygulamak
veri.HP.apply(lambda n : n/2)
# Diğer sütunları kullanarak sütun tanımlama
veri["total_power"] = veri.Attack + veri.Defense
veri.head()
# bizim index ismimiz 'name' 
print(veri.index.name)
# hadi değiştirelim
veri.index.name = "index_ismi"
veri.head()
# İndeksin üstüne yazalım
# if we want to modify index we need to change all of them.
# eğer indexi düzeltmek istiyorsak hepsini değiştirmemiz gerekicek
veri.head()
# verilerimizin ilk kopyasnı data3'e atalım sonra indeks ile rahat rahat oynayabiliriz
veri3 = veri.copy()
# lets make index start from 100. It is not remarkable change but it is just example
# hadi indeksi 100'den başlatalım. Ahım şahım bir değişim değil, sonuçta sadece bir örnek.
veri3.index = range(100,900,1)
veri3.head()
# We can make one of the column as index. I actually did it at the beginning of manipulating data frames with pandas section
# Bir sütunu index olarak yapabiliriz. Biz aslında veri setimize manipüleyi pandas bölümünün başında yaptık
# Nasıl mı şöyle
# veri= veri.set_index("#")
# ayrıca bu da olabilir
# veri.index = veri["#"]
# Hadi bakalım tekrardan veri setimizi baştan başlamak için bir kez daha okuyal
veri = pd.read_csv('../input/pokemon.csv')
veri.head()
# Gördüğünüz gibi bi indeksimiz var. Ancak, daha fazla sütunu indeks olarak ayarlamak istiyoruz
# İndeksi ayarlama: type 1 dış type 2 ise iç indeks
veri1 = veri.set_index(["Type 1","Type 2"]) 
veri1.head(100)
#veri1.loc["Fire","Flying"] # Nasıl kullanılır deneyebilirsin
sozluk = {"tedavi":["A","A","B","B"],"cinsiyet":["K","E","K","E"],"tepki_suresi":[10,45,5,9],"yas":[15,4,72,65]}
df = pd.DataFrame(sozluk)
df
# pivoting
df.pivot(index="tedavi",columns = "cinsiyet",values="tepki_suresi")
df1 = df.set_index(["tedavi","cinsiyet"])
df1
# hadi istiften çıkaralım
# seviye indekslerini belirler
df1.unstack(level=0)
df1.unstack(level=1)
# swaplevel : iç ve dış seviye indeks konumunu değiştme
df2 = df1.swaplevel(0,1)
df2
df
# df.pivot(index="tedavi",columns = "cinsiyet",values="tepki_suresi")
pd.melt(df,id_vars="tedavi",value_vars=["yas","tepki_suresi"])
# df'yi kullanıcaz
df
# tedaviye göre diğer özelliklerin alınması

df.groupby("tedavi").mean()   # ortalama agregasyon / redüksiyon metodudur
# tabiki sum, std, max veya min gibi başka yöntemlerde var
# sadece özelliklerden birini seçebiliriz
df.groupby("tedavi").yas.max() 
# Ya da birden çok özellik seçebiliriz
df.groupby("tedavi")[["yas","tepki_suresi"]].min() 
df.info()
# cinsiyet gördüğünüz gibi bir nesne
# Ancak, groupby kullanırsak, categorical verileri dönüştürebiliriz.
# Çünkü categorical veriler daha az bellek kullandığından, groupby gibi işlemleri hızlandırır
#df["cinsiyet"] = df["cinsiyet"].astype("category")
#df["tedavi"] = df["tedavi"].astype("category")
#df.info()

