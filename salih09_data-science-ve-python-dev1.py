# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/creditcardfraud/creditcard.csv') #Yukarıda bulduğumuz adresi Pandas olarak input etmek (yani bu veriye ulaşmak istediğimizde) soldaki kodu kullanıyoruz.

data.head(10) #Veri setinin ilk X satırının gözlemlemek için data.head kodunu kullanıyoruz. Değer yazmadığımızda ilk 5 satır görününr. Ben burada 10 satırı görmek için parantez içine 10 ifadesini yazdım.
data.info() #Bu kod ile verilerin  tipleri hakkında bilgi sahibi oluyoruz.
data.corr() # Sütunlardaki her verinin kendi içindeki korelasyonu bulmaya çalıştık. Verilerden 0 olanlar ilişki olmadığını, 1'e doğru yaklaştıkça pozitif bir korelasyon olduğunu, -1'e doğru yaklaştıkça negatif korealasyon olduğunu bizlere gösterir.
import matplotlib.pyplot as plt #plt ifadesini önceden tanımlamamıştık. Burada onu tanımlıyoruz. 

import seaborn as sns  # sns'yi de görselleştirme (visualization tool) için kullanıyoruz.

#korelasyonu görsel olarak görüntülemek için aşağıdaki işlemi yapıyoruz.

f,ax = plt.subplots(figsize=(32, 32)) #Parantez içindeki değerler bizlere çizim alanının boyutunu gösteriyor.

sns.heatmap(data.corr(), annot=True, linewidths=.7, fmt= '.1f',ax=ax) #sns kodunu görselleştirme için kullanıyoruz.

plt.show() # En altta formülün çıkmasını engelliyor.
data.columns #Verisetindeki sütunların neler olduğunu göremk için kullanıyoruz.
# Çizgi Grafiği

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.V7.plot(kind = 'line', color = 'b',label = 'V7',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.V8.plot(color = 'r',label = 'V8',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='lower right')     # anahtarın sağ altta olması için kullanıyoruz. 

plt.xlabel('V7')              # adını yazmak için kullanıyoruz.

plt.ylabel('V8 Değeri')

plt.title('Credit Card Fraud')            # Grafiğin başlığını yazıyoruz.

plt.show()
# Dağılım grafiği

data.plot(kind='scatter', x='V7', y='V8',alpha = 0.7,color = 'b') #Alpha, soluklaştırma için kullandığımız değerdir.

plt.xlabel('V7')  # x ekseninin adı 

plt.ylabel('V8')  # y ekseninin adı

plt.title('V7 & V8 Dağılım Grafiği')            # Grafiğin başlığı

plt.show()
# V7 için Histogram grafiğini aşağıdaki formül ile çiziyoruz.

data.V7.plot(kind = 'hist',bins = 1000,figsize = (12,5)) # Değerler 0 ve çevresinde dağıldığı için Bins değerini 1000 yaparak sıklığın dağılımını daha net görebilidik.

plt.show()
#sözlük oluşturma

dictionary = {'elma' : 'apple','üzüm' : 'grape'}

print(dictionary.keys())

print(dictionary.values())

dictionary['elma'] = "apple1" 

print(dictionary)

dictionary['kavun'] = "melon"       # Sözlüğe kavun kelimesini ekliyoruz.

print(dictionary)

del dictionary['elma']              # Sözlükten elma kelimesini siliyoruz. 

print(dictionary)

print('kavun' in dictionary)        # kavun sözlükte vaar mı? Gelecek yanıt true veya false (var ya da yok)

dictionary.clear()                   # sözlüğü temizle

print(dictionary)

del dictionary         # tüm sözlüğü silmek için kullanıyoruz. 

print(dictionary)   # bir üst satırda sözlüğü slidiğimiz için kod hata vermelidir. 

import pandas as pd #Padndas'ı pd olarak import edeceğimizi tanımlıyoruz.

data = pd.read_csv('../input/creditcardfraud/creditcard.csv')  #pd ifadesi Pandas olarak okuma yapmak istediğimizi ifade ediyor.

series = data['V7']        # V7 sütununu seri olarak tanılmadık

print(type(series))

data_frame = data[['V7']]  # V7'yi tablo olarak tanılmadık. Aşağıda bunun tanımlı olduğu formüle ulaşacağız.

print(type(data_frame))
# Kıyaslama operatörü

print(30 > 29)  # 30, 29'dan büyük ise True, değil ise false

print(30!=29)  # 30, 29'a eşit ise True, değil ise false

# Boolean operatörleri (Mantık ile ilgili işlemler)

print(True and False) # True ve False, False döndürür. Örneğin 3<4 ve 6<3 koşululunun beraber sağlanmaması durumunda Talse olur.

print(True or False) # True veya False, True döndürür. Örneğin 3<4 veya 6<3 koşululunun birinin sağlanması durumunda True olur.
# Tablodaki verileri filtreleme. 

x = data['V7']> 40     # V7 değerinin 40'tan büyük olduğu satırları filtreliyoruz. Burada iki satır 40'tan büyük olarak çıktı.

data[x]
# Burada, iki üst kod bölmesinde gördüğümüz Boolean operatörünü (mantık) kullanarak işlem yapacağız. 

data[np.logical_and(data['V7']>40, data['V6']<70 )] #V7 40'tan büyük, V6 ise 70'ten küçük olsun istiyoruz. 
i = 0  # i değerini 0'dan başlatıyoruz. 

while i != 10 :  # i döngüsü 10 sayısına kadar sağlıyoruz. 10 değerine ulaştığında döngüden çıkacak. Mesela buraya 11 yazar isek sayılar 10, 12 şeklinde gideceği ve hiçbir zaman 11 olmayacağı için döngüden çıkmayacaktır.

    print('i: ',i)  # i değerlerinin çıktısını bizlere verecek. 

    i +=2   # Her i değerine 2 ekleteceğiz. İlk değerimiz 0 idi. 0,2,4,6 şeklinde devam eden bir döngü içinde olacağız.

print(i,' döngü sonunda değerimiz 10')  #Döngüden çıktıktan sonraki ifadeyi yazacağız.
# Stay in loop if condition( i is not equal 5) is true

lis = [2,4,6,8,10] # Liste değerlerini tanımladık.

for i in lis:  # i değerlerini listeden çek.

    print('i değeri: ',i) # Listedeki değerlerin karşılığını yaz.

print('')



for index, value in enumerate(lis):  # Burada index değerinin karşılığını buluyoruz. index değeri 0'dan başlar.

    print(index," : ",value)

print('')   



# Sözlük için 

dictionary = {'elma':'apple','kavun':'melon'}

for key,value in dictionary.items():

    print(key," : ",value) # anahtar ve değerleri (karşılığını) yazdıracağız.



dictionary = {'elma':'apple','kavun':'melon'}

for key,value in dictionary.items():

    print(value," : ",key) # değerleri (karşılığını) ve anahtarı yazdıracağız.

  

# Pandas için index değeri 

for index,value in data[['V7']][0:1].iterrows(): #V7'nin 1. satırdaki değerini buluyoruz. Aşağıda 0.239599 olarak yazacak.

    print(index," : ",value) 
def tuple_ex(): #Derste tuble olarak tanımlanmış ama olması gereken tuple (demet) olabilir. Bununla norma parantezli String (karakter dizisi) verileri tanımlanır. 

    t = (4,5,9)

    return t

a,b,c = tuple_ex()

print(a,b,c)
x = 7  # x değerini global scope olarak tanımlandı. Bu en soldadır. 

def f():

    x = 99

    return x   # x değerini local scope olarak tanımlandı. Bu bir tab içeridedir. Local scope altında işlem yaptığımızda bunun sonucunu döndürür.

print(x)      # x = 7 global scope

print(f())    # x = 99 local scope
# local scope yok ise global scope verisi ile işlem yapılır.

x = 50

def f():

    y = 2*x        # local scope x verisi yok

    return y

print(f())         # bu sebeple global scope x verisi kullanılır.
# built in scope ifadesi de Python'un kendi tanımladığı ifadelerdir. Bu ifadelere erişmek için aşağıdaki dormülü kullanabiliriz.

import builtins

dir(builtins)
#nested (fonksiyon içinde fonkisyon)

def kare(): # kare almak için bir fonksiyon yazacağız. Neyin karesini alacağımız aşağıda "return add()**3" ile ifade ediliyor. 

    def toplam ():

        x = 20

        y = -5

        z = x + y # burada da bir toplam formülü tanımladık. 

        return z

    return toplam()**3

print(kare()) 
# default arguments - varsayılan değişkenler.

def f(a, b = 10, c = 8):

    y = a + b + c

    return y

print(f(6))  # Burada b ve c değerleri yukarıdaki fonksiyonda tanımlandığı için aksi belirtilmedikçe 10 ve 8 değerlerini alır.

# Bu değerlerin aksini belitmek istersek istenilen değerler belirilelibir. 

print(f(6,4,1))
# flexible arguments *args

def f(*args):

    for i in args:

        print(i)

f(10)

print("")

f(4,3,2,1)

# flexible arguments **kwargs that is dictionary



def f(**kwargs):

    for key, value in kwargs.items():               # If you do not understand this part turn for loop part and look at dictionary in for loop

        print(key, " ", value)

f(Ülke = 'Türkiye', Başkent = 'Ankara', Nüfus = 83154997)
# Formülleri daha kolay bir şekilde yazmak için lambda fonksiyonu kullanabiliriz. 

kare = lambda x: x**2 

print(kare(4))

toplam = lambda x,y,z: x+y+z

print(toplam(1,2,3))
# Anonymus (liste değeri üzerinden hesaplama yapmak için kullanılır)

liste = [3,6,9]  # Bir liste tanımlıyoruz.

y = map(lambda x:x**3,liste)  #Listedeki değerlerin 3. kuvvetini almak için soldaki kodu yazıyoruz.

print(list(y))
# Iterators

name = "Türkiye"

iterasyon = iter(name)  # iterasyon yapılacak ifadeyi tanımlıyoruz.

print(next(iterasyon))    # ilk değeri yazıyor.

print(*iterasyon)         # kalan değerleri yazıyor.
# zip example

liste1 = [2,6,4,9] 

liste2 = [12,3,5,7]  # iki adet liste tanımlıyoruz. 

zipli = zip(liste1,liste2) #bunları zipli diye bir isimle birleştiriyoruz.

print(zipli)

zlistesi = list(zipli)  # zlistesi adında yeni bir liste tanımlıyoruz. 

print(zlistesi) # tanılmadığımız z listesini yazdırıyoruz. 
un_zip = zip(*zlistesi)  #birleştirdiğimiz zlistesi değerini tekrar ayrımak için bu formülü kullanıyoruz. 

un_liste1,un_liste2 = list(un_zip) # unzip yaptığımızda türü tuple oluyor.

print(un_liste1)

print(un_liste2)

print(type(un_liste2))
# list comprehension

num1 = [6,9,7,8,12,5]  # Liste değerini tanımlıyoruz. 

num2 = [i **2 for i in num1 ]  # Bir üst satırdaki liste değerlerinin karesini alıp, alt satırda yazdırıyoruz.

print(num2)
# iterable (Koşullu)

num1 = [6,20,22]

num2 = [i**2 if i == 20 else i-4 if i < 9 else i+15 for i in num1] # Bu formül şunu anlatır: eğer i değeri 20'ye eşit ise onun karesini al,9'den küçük ise 4 çıkart, hiçbiri değilse 15 ekle

print(num2)
# lets return pokemon csv and make one more list comprehension example

# lets classify pokemons whether they have high or low speed. Our threshold is average speed.

data = pd.read_csv('../input/creditcardfraud/creditcard.csv')

threshold = sum(data.V7)/len(data.V7)  # ortalama almak için bu formülü yazdık

data["V7"] = ["ortalamadan_yüksek" if i > threshold else "ortalamanın_altında" for i in data.V7]
data = pd.read_csv('../input/creditcardfraud/creditcard.csv')

data.head() # İlk 5 satırı gösterir.
data.tail() # Son 5 satırı gösterir.
data.columns # sütun başlıklarını gösterir.
data.shape # Satır ve sütun sayısını gösterir.
data.info() # Veriler hakkında bilgi verir.
# Verilerin sıklığını aşağıdaki formül ile buluyoruz. 

print(data['V7'].value_counts(dropna =False))  # eksik veriler olsa bunu da bulacaktık. Ancak ekssik veri yok. 
data.describe() #istatistiksel veriler (ortalama, medyan vb.) bulmak için bu formülü kullanıyoruz.
# Black line at top is max

# Blue line at top is 75%

# Red line is median (50%)

# Blue line at bottom is 25%

# Black line at bottom is min

# There are no outliers

data.boxplot(column='V7') #V7 değerlerini histogram olarak çizdiriyoruz. 
# Verisetinin ilk on satırından yeni bir veri tanımlıyoruz.

data_yeni = data.head(10)    # I only take 5 rows into new data

data_yeni
# Veriyi daratma. Yukarıda 10 satıra daraltmıştık. Burada ise Amount ve Class sütunları ile daraltıyoruz. Bunları değişken olarak alt alta ekliyor.

melted = pd.melt(frame=data_yeni,id_vars = 'V7', value_vars= ['Amount','Class'])

melted
# Melting işleminin tersini (Reverse) yapıyoruz.

# I want to make that columns are variable

# Finally values in columns are value

melted.pivot(index = 'V7', columns = 'variable',values='value')
# Veri birleştirme (Concentrating Data) Veri setinin ilk ve son 5 satırını birleştirdik.

data1 = data.head()

data2= data.tail()

conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # a

conc_data_row
data1 = data['V7'].head()

data2= data['Amount'].head()

conc_data_col = pd.concat([data1,data2],axis =1) # V7 ve Amount sütunlarını birleştiriyoruz. 

conc_data_col
#5 Temel veri tipi (data types): object(string),booleab, integer, float and categorical.

data.dtypes #Tipleri bu formül ile bluuyoruz.
data['V7'] = data['V7'].astype('category')

data['Amount'] = data['Amount'].astype('int')

data.dtypes #Türleri değiştirip tekrar veri tiplerini sorguluyoruz.
data.info() #Null veri bulunmuyor. Veri seti dolu.
data["V7"].value_counts(dropna =False) #NaN value yok. 
# data frames from dictionary

import pandas as pd # Bu satırı yazmadığımızda "pd" nin ne olduğunu tanıyamayacaktır. 

country = ["Türkiye","Almanya"]

population = ["83","82"]

list_label = ["Ülke","Nüfus"]

list_col = [country,population]

zipped = list(zip(list_label,list_col))

data_dict = dict(zipped)

df = pd.DataFrame(data_dict)

df

# Yeni sütun eklemek için

import pandas as pd

df["Başkent"] = ["Ankara","Berlin"]

df
# Tüm sütuna aynı değeri tanımlamak için - Broadcasting

df["Gelir"] = 0

df
# Tüm verileri Plot çizdirme

data = pd.read_csv('../input/creditcardfraud/creditcard.csv')

data1 = data.loc[:,["V7","V8","V9"]]

data1.plot()
import matplotlib.pyplot as plt # Tüm verileri ayrı grafiklerde çizdiriyoruz.

data1.plot(subplots = True)

plt.show()
# scatter plot 

data1.plot(kind = "scatter",x = "V7",y= "V8")

plt.show()
# hist plot

data = pd.read_csv('../input/creditcardfraud/creditcard.csv')

data1.plot(kind = "hist",y = "V7",bins = 1000,range= (-10,10))

# histogram çizimlerini normal ve kümülatif olarak çizdiriyoruz.

fig, axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind = "hist",y = "V7",bins = 1000,range= (-10,10),ax = axes[0])

data1.plot(kind = "hist",y = "V8",bins = 1000,range= (-10,10),ax = axes[1],cumulative = True)

plt.savefig('graph.png')

plt
data.describe()
time_list = ["1992-03-08","1992-04-12"]

print(type(time_list[1])) # As you can see date is string

# however we want it to be datetime object

datetime_object = pd.to_datetime(time_list)

print(type(datetime_object))

import warnings #Uyarıları kapatmak için bu komutu kullanabiliriz. 

warnings.filterwarnings("ignore") #Uyarıları kapatmak için bu komutu kullanabiliriz. 

# In order to practice lets take head of pokemon data and add it a time list

data2 = data.head() #Veri seninin ilk 5 satırını alıyoruz. 

date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"] # Buraya 5 adet tarih atadık.

datetime_object = pd.to_datetime(date_list) 

data2["date"] = datetime_object #Liste halindeki objeleri dataset'e ekliyoruz.

data2= data2.set_index("date") # Bu veriyi index yapmak için bu formülü kullanıyoruz. 

data2 
# Tarihlere göre verileri buluyoruz.

print(data2.loc["1992-03-10":"1993-03-16"])
# Yukarıdaki data2 verisini kullanıyoruz.

data2.resample("A").mean() #A ifadesi Annual - yıllık, M olsa idi Monthly (aylık) olacaktı.
# Ay ile dönüşüm yapmak istersek

data2.resample("M").mean()

# Tüm aylara ait veriler bulunmadığı (tanımlamadığımız) için NaN verileri ile karşılaşıyoruz.
# NaN olan verileri interpolasyon ile doldurabiliriz. Bunu da aşağıdaki formül ile yapacğaız. 

data2.resample("M").first().interpolate("linear")
# Ortalama değerleri alarak da interpolasyon yapmak istersek aşağıdaki formülü kullanabiliriz.

data2.resample("M").mean().interpolate("linear")
data = pd.read_csv('../input/creditcardfraud/creditcard.csv')

data.head()

data.set_index('Time')

data.head()

# indexing using square brackets

data["V5"][4]  #V5 sünunun 4. index verisini buluyoruz.
data.loc[1,["V7"]]
# Bazı sütunları seçmek için

data[["V7","Amount"]]
# Seriler ve Data Frame'ler arasındaki fark aşağıda tanımlanmıştır.

print(type(data["V7"]))     # seriler

print(type(data[["V7"]]))   # data frame ler
# Slicing and indexing series

data.loc[0:5,"V7":"Amount"]   # V7'den, Amount üstununa kadar, 0. index'ten, 5. index'e kadar verileri seç
# Aşağıdan yukarı sıralama

data.loc[5:0:-1,"V7":"Amount"] 
# Bir yerden, sona kadar slicing 

data.loc[0:5,"Amount":] 
# boolean serileri oluşturma

boolean = data.V7 > 35 #V7'nin 35'ten büyük olduğu seriyi oluşturuyoruz.

data[boolean]
first_filter = data.V6 > 23

second_filter = data.V8 > -10

data[first_filter & second_filter]  # İki filtreyi de uygulayarak bulduğumuz sonucu yazdırıyoruz.
# Filtering column based others V8, -20'den büyük iken V7 verilerini yazdır 

data.V7[data.V8>-20]
# Plain python functions

def div(n):

    return n/3 # 3'e bölmek için bir fonksiyon tanımlıyoruz. V7 sütununa bu fonksiyonun tanımlanmasını yazdırıyoruz.

data.V7.apply(div)
# Or we can use lambda function

data.V7.apply(lambda n : n/3)
# Defining column using other columns

data["V7&V8"] = data.V7 * data.V8 # Verileri kullanarak başka bir sütuna atama yapıyoruz.

data.head()
# index adını aşağıdaki formül ile çalıştırıyoruz. İlk etapta ismi olmadığı için none çıkıyor.

data = pd.read_csv('../input/creditcardfraud/creditcard.csv')

print(data.index.name)

data.index.name = "endeks" # index adını endeks olarak tanımlamak için bu formülü kullanıyoruz. 

data.head()
# Overwrite index

# if we want to modify index we need to change all of them.

data.head()

# first copy of our data to data3 then change index 

data3 = data.copy()

# index değerlerini 100'den başlatıp 284907'ye kadar 1'er 1'er artırdık

data3.index = range(100,284907,1) 

data3.head()
data = pd.read_csv('../input/creditcardfraud/creditcard.csv')

data.head(20)
# Time index : Class is outer Time is inner index

data1 = data.set_index(["Class","Time"]) 

data1.head(100)

# data1.loc["Fire","Flying"] # how to use indexes
dic = {"spor":["Basket","Basket","Futbol", "Futbol"],"Cinsiyet":["K","E","K","E"],"Yaş":[22,18,25,33],"Başarı":[11,4,8,24]}

df = pd.DataFrame(dic)

df
# pivoting

df.pivot(index= "spor",columns = "Cinsiyet", values= "Başarı")
df1 = df.set_index(["spor","Cinsiyet"])

df1

# unstack
# level determines indexes

df1.unstack(level=0)
df1.unstack(level=1)
# change inner and outer level index position

df2 = df1.swaplevel(0,1)

df2
df
# df.pivot(index="treatment",columns = "gender",values="response")

pd.melt(df,id_vars="Cinsiyet",value_vars=["Yaş","spor"])
df
# according to treatment take means of other features

df.groupby("spor").mean()   # mean is aggregation / reduction method Spor değerlerinin ortalamasını alıyoruz

# there are other methods like sum, std,max or min
# we can only choose one of the feature

df.groupby("spor").Yaş.max() 
# Or we can choose multiple features

df.groupby("spor")[["Yaş","Başarı"]].min() 
df.info()

# as you can see gender is object

# However if we use groupby, we can convert it categorical data. 

# Because categorical data uses less memory, speed up operations like groupby

#df["Yaş"] = df["Cinsiyet"].astype("category")

#df["spor"] = df["spor"].astype("category")

#df.info()