# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns #visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataframe = pd.read_csv("../input/pokemon.csv")
dataframe.info()
#correlation map-->bu bize feature arasındaki ilişkiyi verir.Bunun için corr() metodunu kullanırız.ama görselleştirme için seaborn kütüphanesi
#korelasyon 0-1 arası dogru orantı gücünü gösterir. 0sa ilişki yoktur. -1 ve 0 arası ters orantıyı gösterir!
dataframe.corr()  #gördüğümüz gibi excel tablosu cıktı.

f,ax=plt.subplots(figsize=(18,18)) #yani f burda figure dur.bu tamamen çıkan görselin boyutunu belirler.18e 18
sns.heatmap(dataframe.corr(),annot=True,linewidth=.8,fmt=".1f",ax=ax) #görsel için seaborn kütüphanesinin heatmap() metodu kullanılır.
#data.corr() tabloyu alır,annot=True korelasyon sayılarının gözükmesi demek,linewidth çizgi kalınlıgı,fmt= virgülden sonraki basamak sayısı
plt.show()
dataframe.corr().unstack().sort_values().drop_duplicates()  #bu da korelasyonları sıralayan bir kod!
dataframe.head(10)
dataframe.columns
fig = plt.subplots(figsize=(15,10))
dataframe.Speed.plot(kind="line",color="green",alpha=1,label="Speed",grid=True,linestyle=":")  #burdaki kind plot ın çeşidini söyler.line mı scatter mı
dataframe.Defense.plot(color="red",label="Defense",linestyle="-.",linewidth=1,alpha=0.5,grid=True)#gridi ikisinede yazmalıyız.
plt.legend(loc="upper right") #lejantın yerini böyle belirtebiliyoruz.
plt.title("Line Plot")
plt.show()
#scatter plot-->iki değer arasındaki korelasyonadu.attack ve defense arasındaki korelasyona bakalım.
#burda bir köşegen oluşuyorsa renklerle doğru orantılı bir korelasyon var diyebiliriz sezgisel olarak
dataframe.plot(kind="scatter",x="Attack",y="Defense",color="blue",alpha=0.5) #burda x ve y eksenine feature bir ismi veriyoruz.
plt.title("Scatter Plot")
plt.show()
#Diğer bir şekilde şöyle oluyordu.
plt.scatter(dataframe.Attack,dataframe.Defense,color="grey")  #ilk x ekseni sonra y ekseni
plt.show()
#histogram
dataframe.Attack.plot(kind="hist",figsize=(10,10),color="purple",bins=20)
plt.show()
plt.hist(dataframe.Defense,color="g")
plt.show()
#x ekseni ve y ekseninde 0 ,25 ,50 ,75 ve 0,10,20,30 şeklinde olan rakamları 
#örneğin 0,5,10,15,20,25 ve 0,5,10,15,20 şeklinde daha da küçültmek istersek
dataframe.Attack.plot(kind="hist",figsize=(10,10),color="purple",bins=20,grid=True)  #bu figsize metodu sadece kind olarak kullandıgımız kodlarda!
plt.xticks(np.arange(0,200,10))  #ilk değer,sondeğer,step
plt.show()
#plot u eğer clean etmek istersek clf() metodunu kullancaz!!
plt.hist(dataframe.Defense,color="g")
plt.clf()
#plt.subplot(figsize= [x,y]) metodu ile grafiklerin boyutunu büyütebiliriz!
#Deep learning te cok kullanacağız dict leri
dictionary = {"spain":"madrid","usa":"vegas"}
print(dictionary.keys()) 
print(dictionary.values())
dictionary["spain"]="barcelona" #update existing entry
dictionary
dictionary["france"]="paris" #update new key and value
dictionary
del dictionary["spain"] #remove existing entry
dictionary
print("france" in dictionary)  #check out entry
dictionary.clear() #remove all entry in dict
dictionary
#dictionary içini temizledik ama hala hafıza da boş olsa da duruyor onun için
del dictionary
print(dictionary)
data = pd.read_csv("../input/pokemon.csv")  #csv dosyasını dataframe yapma
data
#Pandasta iki çeşit datatype biri series diğeri dataframedir.
#series vektör gibi tek yönlüdürler.
#dataframe ise iki boyutludur.Bundan dolayı kullandıkları metodları da farklıdır.
series = data["Defense"]
print(type(series))
data_frame = data[["Defense"]]
print(type(data_frame))
filtre = data["Defense"]>200
data[filtre]
#Filtering pandas with logical_and(np kütüphanesi)
#hem defansı 200ten büyük hem de attack ı 100 den büyük
x=data["Defense"]>200
y=data["Attack"]>100
data[x&y]  #bu bizim gördüğümüz yoldu.
#Eğer 3 koşullu olursa da böyle yapıyoruz.
z=data["HP"]>70
data[x&y&z]
#numpy kütüphanesini kullanırsak böyle olur.Ama üçlü kullanamıyoruz burda!
data[np.logical_and(data["Defense"]>200,data["Attack"]>100)]
i=0
while i != 5:
    print("i is : ",i)
    i+=1
print("i is -- 5")
# biz enumarate edebiliriz listenin elemanlarını index leri ile yazdırabilirz.
lis = [1,2,3,4,5]
for i,j in enumerate(lis): #bu machine learning ve deep learning yaparken değer kadar index i de önemlidir!
    print(i,":",j)
dictionary = {"spain":"madrid","france":"paris","turkey":"istanbul","turkey":"istanbl"} #sözlükte bir key bir defa sayılır.
for i,j in dictionary.items():  #items metodu hem key hem value değerini döndürür.
    print(i,":",j)
#biz bu for döngüsünü data mızda yapmak istersek index le beraber yine enumarate metodunu kullanabiliyoruz!
for i,j in enumerate(data["Attack"][0:4]):
    print(i,":",j)

#ama iterrows() metodunu da kullanabiliriz!
for i,j in data["Defense"][1:11].iterrows():
    print(i,":",j)
#Görüldüğü gibi cok güzel bir hata aldık diyor ki series de diyor iterrows() metodu kullanılmaz.
#Yani dataFrame lerde kullanılır.DataFrame içinde iki tane köşeli parantez kullanıyorduk(2 boyutlu ya hani)7
#Bana sorarsan enumerate cok daha iyi!
for i,j in data[["Defense"]][1:11].iterrows():
    print(i,":",j)
def tuble():
    t=(3,42)
    return t
tuble()
a,b=tuble()
print(a,_)
x=2
def f ():
    x=3
    return x
print(x)   #main body de tanımladıgımız x ile funcktionda tanımladıgımız farklı x oluyo
print(f())
#eğer fonks. içinde local x yoksa global e bakıyo yani main body ye
x=5
def f():
    y=2*x
    return y
print(f())
#iç içe fonksiyonlarda böyle kullanılıyo
def square(a):
    y=a*2
    def add():
        x=4
        z=y+x
        return z
    return add()**2
print(square(3))
def f(a,b=3,c=2):
    y=a+b+c
    return y
print(f(13))
print(f(13,23,0))
def f(*args):
    toplam=0
    for i in args:
        toplam+=i
    return toplam
print(f(2,3,4,5,6,7))
def f(**kwargs):
    for key,value in kwargs.items():
        print(key," ",value)
f(country="ıtaly",favor="Venedik",population=123421)
liste=[1,2,3]
y=map(lambda x:x**2,liste)
print(list(y))
y=[i**2 for i in liste  if i>1] #buda list comprehension
print(y)
y=[i**2 if i>1 else i-2 for i in liste] #if else kullanacaksak da böyle
print(y)
name="furkan"
it = iter(name)
print(next(it)) #her bir elemanı sırası ile yazdırır
print(*it) # bu da geri kalan kısmı yazdır demek!
print(it)
print(next(it)) # hepsini döndürdüğümüz için bi daha next i kullanamıyoruz.
liste=range(1,6)
liste1=range(6,11)
liste2=list()

a=zip(liste,liste1)
print(list(a))  #bu zip fonksiyonunu da for döngüsüz kullancaksak list methodu ile yazdırabiliriz!

for i,j in zip(liste,liste1):
    liste2.append((i,j))
print(liste2)
#Eğer biz unzip yapmak istersek
liste=[(1,2),(2,3),(3,4),(4,5)]
unzip=zip(*liste)
un_list1,un_list2=list(unzip)
print(un_list1)
print(un_list2)  #bunlar gördüğümüz tuple a ceviriyor python.istersek listeye cevirebiliriz.
num1=[5,10,15]
num2=[i**2 if i==10 else i-5 if i>7 else i+5 for i in num1]
#burda şöyle i=10 ise i**2 else if dediğimiz elif oluyo yani i>7 ise i-5 sonra else ne 10a eşit ne de
#7den büyük değilse i+5 yap demek!
print(num2)
#Datamıza dönersek
x_bar=np.mean(dataframe.Speed)  #numpy ile mean
print(x_bar)
dataframe["Speed_Level"]=["High" if i>x_bar else "Low" for i in dataframe.Speed]
dataframe.loc[2:10,["Speed","Speed_Level"]] #diyo ki 2.satırdan 10.satıra kadar sadece bu iki sütun
dataset=pd.read_csv("../input/pokemon.csv")
dataset.head()
#görüldüğü üzere büyük harf küçük harf prob.,kelime arası boşluk,nan(missing data) bunlar var.
dataset.tail()
dataset.columns
dataset.shape  #bu 800tane satır ve 12 tane feature var!
dataset.info()
#Count Frequency
#burda water type ından 112 tane pokemon varmıs. ve ya grass typeından 70 tane pokemon varmış
print(dataset["Type 1"].value_counts(dropna=False))
print(dataset["Type 2"].value_counts(dropna=False)) #dropna=False demek, sayılacak nan değerleri varsa say ve yazdır
                                                    #dropna=True demekse nan varsa bile yazdırma 
#bu method boş olan değerleri ignore eder!
dataset.describe() #bize sadece nümerik değerler için bu tabloyu verir
dataset.boxplot(column="Attack",by="Legendary")
plt.show()
#görüldüğü gibi outlierlar gösterilmiş. Attack legandary e göre boxplot yapılmış
#acaba bu outlier lar dogrumu dogruysa da baska bunun gibi değerler olsa datam nasıl değişir.
dataset.boxplot(column="Attack")
plt.show()
data_new=dataset.head()
data_new
#Biz bu melt fonksiyonunu şu yüzden kullanıyoruz.seaborn kütüphanesindeki tool ları kullanabilmek için!!
#pandas ve seaborn arasındaki köprü gibi
#burda melt fonksiyonu içinde yazan id_vars,frame,value_vars bunlar metodun tanımlamalarıdır bizim 
#kafamıza göre değil
#frame=hangi datayı melt yapcaz,id_vars=hangi feature oldugu gibi kalacak,value_vars=variable ve value 
#hangilerini alacak

melted_data=pd.melt(frame=data_new,id_vars="Name",value_vars=["Attack","Speed","Defense"])
melted_data
#burda da index,columns,values metodun kendi tanımlamaları
melted_data.pivot(index="Name",columns="variable",values="value")

data1=dataset.head()
data2=dataset.tail()
conc_data=pd.concat([data1,data2],axis=0,ignore_index=True) 
#axis=0 adds dataframe s in row yani vertical seklinde yukarıdan asagı
#ignore_index=True tanımı da indexleri ignore et sen sırasıyla kendin index ata
conc_data
conc_data=pd.concat([data1,data2],axis=0,ignore_index=False)
#False yaparsak kendi indeksini yazar!
conc_data
#bide horizontal concat yaparsak
data1=dataset.Name.head()
data2=dataset.Attack.head()
data3=dataset.Defense.head()
hor_data=pd.concat([data1,data2,data3],axis=1,ignore_index=True)
hor_data
dataset.dtypes
dataset["Type 1"]=dataset["Type 1"].astype("category")
dataset["Attack"]=dataset["Attack"].astype("float")
dataset.dtypes
dataset.head()
dataset.info()
#Type 2 deki sıkıntıya bakalım.nan mı yoksa boslukmu
dataset["Type 2"].value_counts(dropna=False) #nanları drop etme false yani
data=dataset
data["Type 2"].dropna(inplace = True)  # inplace = True means we do not assign it to new variable. 
#Changes automatically assigned to data
data.head(10)
data["Type 2"].value_counts(dropna =False) #gördüğümüz gibi Nan ları drop etti.

assert 1==1 #1 1e eşitmidir eğer eşitse hata vermez
assert 1==2  #eğer yanlıssa görüldüğü gibi hata veriyor.
assert data["Type 2"].notnull().all() #diyoruz ki burda type 2 nin hepsi boş değil dimi
data["Type 2"].fillna("empty",inplace=True)  #diyoz ki boş olan type 2 leri "empty" ile doldur
assert data["Type 2"].notnull().all() #diyoruz ki burda type 2 nin hepsi boş değil dimi
data["Type 2"].value_counts(dropna=False)
assert data.columns[1]=="Name" #datanın columnlarının 1.si name midir
assert data.dtypes["Attack"]==np.int  #çünkü float
assert data.dtypes["Attack"]==np.float
#şimdi listeden sözlük sözlükten de bir dataframe oluşturcaz.
country = ["spain","ıtaly"]
population = [123456,234567]
column=["country","population"]
listed=[country,population]
print(listed)
zipped=list(zip(column,listed))
print(zipped)
zipped=dict(zipped)
df=pd.DataFrame(zipped)
df
#add new columns
df["capital"]=["madrid","roma"]
df
#broadcasting
df["income"]=0  #burda tüm satırlara aynı değeri atar
df
data=dataset.loc[:,["Attack","Defense","Speed","Legendary"]]
data.plot() #görüldüğü gibi çok karısık ben bunun için subplot kullanabilirim.
#subplots
data.plot(subplots=True)
plt.show()
#scatter plot
data.plot(kind="scatter",x="Attack",y="Speed")
plt.show()
#hist
data.plot(kind="hist",y="Defense",bins=30,range=(0,300),normed=True)
#normed normalize etmek demek frequency kısmını yüzdelik gibi olasılıksal veriyor
plt.show()
#burda önce axes i tanımlamak lazım 3satır 1 sutun demek
fig, axes = plt.subplots(nrows=3,ncols=1) #burda 3e 2 yazarzak 6 tane kutucuk yapıyo ama biz 
#6 tane tanımlamadık ondan hata veriyor
data.plot(kind="hist",y="Defense",bins=30,range=(0,300),normed=True,ax=axes[0])
data.plot(kind="hist",y="Defense",bins=30,range=(0,300),normed=True,ax=axes[1],cumulative=True)
data.plot(kind="scatter",x="Attack",y="Defense",ax=axes[2])
plt.show()
#Eğer 4 tane tanımlarsak 2 ye 2
fig, axes = plt.subplots(nrows=2,ncols=2) 
data.plot(kind="hist",y="Defense",bins=30,range=(0,300),ax=axes[0,0])
data.plot(kind="hist",y="Defense",bins=30,range=(0,300),normed=True,ax=axes[0,1],cumulative=True)
data.plot(kind="scatter",x="Attack",y="Defense",ax=axes[1,0])
data.plot(kind="scatter",x="Attack",y="Speed",ax=axes[1,1])
plt.show()
#pd.to_datetime()
time_list=["1995-12-31","2005-04-13"]
print(type(time_list[1]))
data_time=pd.to_datetime(time_list) #pandas ile datetime a cevirme
print(type(data_time))
data_time
#bizim pokemon datamızda zamanla alakalı bir feature yoktu şimdi o feature ı ekliyecem ve 
#index sıralamasını zamana göre yapcagım
data=dataset.head()

time=["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"] #ama burda datayı sıralanmıs vermek
#lazım cünkü kendisi otomatik sıralamıyor
d_time=pd.to_datetime(time) #convert to datatime a
data["date"]=d_time

data=data.set_index("date") #indeksi date e göre sıralama
data
#indeximiz time series oldugu için artık loc ve slicing işlemlerini tarih ile yapcaz
print(data.loc["1993-03-16"])
data.loc["1990-03-10":"1994-03-16"] #ordaki tam tarih olmasına da gerek yok
#mesela ben yıllara göre datamın featurelarının ortalamasını alabilirim.
data.resample("A").mean() #gördüğümüz gibi 1992 ve 1993 yıllarının nümerik değerlerinin orta. verdi
#Eğer aylara göre resample yapıp featurların ortalamasını almak istersek
data.resample("M").mean()
#gördüğümüz gibi olmayan aylar nan şeklinde yazılmış.
#Eğer ben bu nan ları doldurmak istersem.Tabi sadece nümerik değerleri
data.resample("M").first().interpolate("linear")
#interpolate("linear") doldur demek nandan önceki değer ile nanın bitişi arasındaki featurların 
#değerlerini doğrusal olarak eşit bi şekilde doldur demek

data.resample("M").mean().interpolate("linear")
#meanleri aynı olacak sekilde interpolate ediyor
#indeksi biz 0 dan değil de 1den baslatmak istersek ve veri setinde bir böyle bir sıralama varsa
data=dataset.head(20)
data=data.set_index("#")
data
data["HP"][1] #artık indeks birden baslıyor
data.HP[1]
print(data.loc[1,"HP"]) #1.satırın Hp değerini bana yaz 
data[["HP","Attack"]] #dikkat çift köşeli parantez iki şey istediğmiz için bir tane daha kullanıyoz
# Difference between selecting columns: series and dataframes
print(type(data["HP"]))     # series
print(type(data[["HP"]]))   # data frames
data.loc[1:10,"HP":"Speed"]  #virgül öncesi satır virgül sonrası sutun feature yani
#Reverse slicing
data.loc[10:1:-1,"HP":"Defense"]
data.loc[1:5,"HP":] #mesela bir featurdan son feature kadar derken : koyup bitirince oluyo
filtre=data.HP > 70
data[filtre]
#iki filtreyi birleştime
filtre=data.HP > 70
filtre2 = data.Speed >80
data[filtre & filtre2]
# Filtering column based others-- mesela hızları 80 den büyük olanların canlarını yazdırmak
data.HP[data.Speed>80]
#apply()
#yazdıgımız bir fonksiyonu bir feature a veya datasete entegre
def div(n):
    return n/2
data.HP.apply(div)
#ve ya lambda ile de olur
data.HP.apply(lambda n:n/2)
#mesela yani bir feature yaratırken diğer feature ları kullanabilirz
data["total_power"]=data.Attack + data.Defense
data.head()
#biz index imizin neye baglı oldugunu şöyle görebiliriz
print(data.index.name)
#biz bunu değiştirmek istersek de
data.index.name="index"
data.head()
