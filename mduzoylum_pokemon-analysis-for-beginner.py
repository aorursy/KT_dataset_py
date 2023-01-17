# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/pokemon-challenge/pokemon.csv')
data.head()
#İlk 5 data yı gösterir
data.info()
# Data ile ilgili tür bilgilerini verir
#correlation
data.corr()
#Datalar arasındaki ilişiki oranını göstermektedir.+1 0 -1
#correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data.head(10)
data.columns
#Kolon isimlerini göstermek için
# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.Speed.plot(kind = 'line', color = 'g',label = 'Speed',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.Defense.plot(color = 'r',label = 'Defense',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x ekseni - Pokemonlar')              # label = name of label
plt.ylabel('y ekseni')
plt.title('Line Plot')            # title = title of plot
plt.show()
#attack ile defense arasındaki ilişkiyi öğrenmek istiyoruz
plt.scatter(data.Attack,data.Defense,color="red",alpha=0.5)
plt.show()
#İlişkiyi görmek için scatter fonksiyonunun farklı yazılış stili
data.plot(kind='scatter', x='Attack', y='Defense',alpha = 0.5,color = 'red')
plt.xlabel('Attack')              # label = name of label
plt.ylabel('Defence')
plt.title('Attack Defense Scatter Plot')            # title = title of plot
plt.show()
# Histogram
# bins = number of bar in figure
data.Speed.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.xlabel("Pokemanların Hızı")
plt.ylabel("Pokemonların Adeti")
plt.show()
data.Speed.plot(kind='hist',bins=100)
plt.clf()
#plt.clf() histogramı sildirmek için
#create dictionary and look its keys and values(Listelerden daha hızlıdır)
dictionary = {'spain' : 'madrid','usa' : 'vegas'}
print(dictionary.keys())
print(dictionary.values())
# Keys have to be immutable objects like string, boolean, float, integer or tubles
# List is not immutable
# Keys are unique
dictionary['spain'] = "barcelona"    # update existing entry
print(dictionary)
dictionary['france'] = "paris"       # Add new entry
print(dictionary)
del dictionary['spain']              # remove entry with key 'spain'
print(dictionary)
print('france' in dictionary)        # İçerisinde france var mı ?
dictionary.clear()                   # remove all entries in dict
print(dictionary)

#hafızadan tamamen sil
del dictionary
series = data['Defense']        # data['Defense'] = series
print(type(series))
data_frame = data[['Defense']]  # data[['Defense']] = data frame
print(type(data_frame))
#Defense özelliği 200 den büyük olanlar
x = data['Defense']>200
data[x] # Datanın içerisindeki sadece true olanları yazdır
data['Defense']>200 # Data içerisindeki her bir değeri true false olarak ayırır
data[np.logical_and(data['Defense']>200,data['Attack']>100)] # Defense 200 den büyük ve Attack 100 den büyük olan değerleri getir
data[(data['Defense']>200) & (data['Attack']>100)] # Defense 200 den büyük ve Attack 100 den büyük olan değerleri getir(Farklı yol)
i=0
while i != 5 :
    print('i is ',i)
    i +=1
print(i,' is equal to 5')
lis = [1,2,3,4,5]
for i in lis :
    print('i is ',i)
print('sona geldin')
for index, value in enumerate(lis) : # index ve value yazdırmak için enumerate kullanılır
    print(index,' : ',value)
print("Sona geldin")
dictionary = {"Spain":"Madrid","France":"Parice"}
for index, value in dictionary.items():
    print(index," : ",value)
for index,value in data[['Attack']][0:1].iterrows():
    print(index," : ",value)
# example of what we learn above
def tuble_ex():
    """ return defined t tuble"""
    t = (1,2,3)
    return t
a,b,c = tuble_ex()
print(a,b,c)
x=2

def f():
    x=3
    return x
print(x)
print(f())
def squar():
    def proces():
        x=3
        y=5
        return x+y
    return proces()**2
squar()
def f(*args):
    print(args)
f(2,3,4,5)
def dic(**args):
    for index,value in args.items():
        print(index,"-",value)

dic(ilce="Meram",il="Konya")
def square(x): # uzun yol
    print(x**2)
square(2)

square2=lambda x:x**2 #kısa yol x->parametre
square2(2)
numberlist=[1,2,3,4]

sonuc=map(lambda x:x**2,numberlist)
print(list(sonuc))
list1=[1,2,3,4,5]
list2=[10,20,30,40,50]
z=zip(list1,list2)
print(z)
z_list=list(z)
print(z_list)
unlistzip=zip(*z_list)
unlist1,unlist2=list(unlistzip)
print(unlist1)
print(unlist2)
print(type(unlist1))
print(type(list(unlist1)))
print(list(unlist1))

listcomp=[1,2,3,4,5,6]
listlast=[i+1 for i in listcomp]
print(listlast)
num1=[7,8,9]
num2=[i+1 if i == 8 else i + 2 if i == 9 else i+3 for i in num1]
print(num2)
ortalama=sum(data["Speed"])/len(data["Speed"])
data["speedlevel"]=["Yüksek" if i > ortalama else "Orta" if i==ortalama else "Düsük" for i in data.Speed]
data.loc[0:100,["speedlevel","Speed"]]
def y(x):
    global a
    a=4
    return 0
def f(a):
    a=3
    print(a)
    return a
a=5
f(a)
print(a)
y(a)
print(a)
a=[[1,2],[2,1]]
b=[[4,1],[2,2]]
np.cross(a,b)
def c(x):
    result=0
    temp=1
    
    for i in range(1,x):
        temp=temp*i
        for k in range(i):
            result+=temp
    return result
    
c(4)
username = input("Enter username:")
print("Username is: " + username)
data.head()

data.columns
data.shape
data.info()
data.tail() # Son 5 değeri gösterir
print(data["Type 1"].value_counts(dropna =False)) # if there are nan values that also be counted
data.describe() # tüm parametreleri göstermektedir.
data.boxplot(column="Attack",by="Legendary")
plt.show()
data_new =  data.head()
data_new
melted=pd.melt(frame=data_new,id_vars="Name",value_vars=['Attack','Defense']) # Belirlenen Kolon ve değerlerin alınması
melted
melted.pivot(index="Name",columns="variable",values='value') # melted edilen data nın tekrar eski haline getirilmesi
data1=data.head()
data2=data.tail()
conc_data=pd.concat([data1,data2],axis=0,ignore_index=True) # concat iki dataframe i birleştirme, axis=0->yatayda birleştir
conc_data
data_attack=data['Attack'].head()
data_defense=data['Defense'].head()
data_at_def=pd.concat([data_attack,data_defense],axis=1) # axis=1->dikeyde birleştir
data_at_def
data.dtypes
data["Type 1"]=data["Type 1"].astype('category') #tip dönüşümleri için kullanılmaktadır
data["Speed"]=data["Speed"].astype('float')
data.dtypes
data.head(10)
data.info()
data["Type 2"].value_counts(dropna=False) # 386 adet değerimiz boş yani NAN gelmekte oldugunu tespit ettik
data1=data
data1["Type 2"].dropna(inplace=True) # dropna->null olanları drop et yani at. inplace=True->çıkardığın değerleri data1 e kaydet
assert 1==1 # Değer doğru ise sonuc döndürmez ama yanlış ise hata verir kotnrol etme fonksiyonu
assert 1==2 # hata verir
data["Type 2"].notnull() # null değil mi her biri için kontrol et?
data["Type 2"].notnull().all() # tümünün değeri true ise tek olarak true döner
assert data["Type 2"].notnull().all() # true oldugu için herhangi birşey yazmadı
data["Type 2"].fillna('empty',inplace=True) # null olanları empty yap

#assert data.columns[1]=="Name" # 1. column un adı Name mi kontrol et?
assert data.Speed.dtypes==np.int #hata verir çünkü Speed in türü float
country=["Spain","Turkey"]
population=["10","20"]
list_label=["country","population"]
list_col=[country,population]
#zipped=list((list_label,list_col)) # zip olmasaydı ->                    [['country', 'population'], [['Spain', 'Turkey'], ['10', '20']]]
zipped=list(zip(list_label,list_col)) # sıkıştır ve liste haline getir -> [('country', ['Spain', 'Turkey']), ('population', ['10', '20'])]
data_dict=dict(zipped) # sözlük şekline dönüştür ->                       {'country': ['Spain', 'Turkey'], 'population': ['10', '20']}
df=pd.DataFrame(data_dict)
df
df["capital"]=["madrid","ankara"] # capital adında yeni bir alan ekledik ve değerlerini yazdık
df
df["income"]=0
df
data1=data.loc[:,["Attack","Defense","Speed"]] # hepsimi tek bir grafikte gösterme
data1.plot() # siyah yazıyı göstermemek için kullanılır
data1.plot(subplots=True) # he birini ayrı grafikte gösterme
plt.show()
data1.plot(kind="scatter",x="Attack",y="Defense") #scatter gösterimi
plt.show()
data1.plot(kind="hist",y="Defense",bins=50,range=(0,500),normed=True) # bins->cubuk kalınlıgı,range->defense deger aralıgı,normed->sayılar normalize edilsin mi
# histogram subplot with non cumulative and cumulative
fig, axes = plt.subplots(nrows=2,ncols=1)
data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[0])
data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)
plt.savefig('graph.png')
plt
time_list=["2020-03-22","2020-03-23"]
print(type(time_list[1]))

datetime_object=pd.to_datetime(time_list)
print(type(datetime_object))
import warnings
warnings.filterwarnings("ignore") # uyarıları engelle

data2=data.head()
date_list=["2020-03-18","2020-03-19","2020-03-20","2021-03-21","2021-03-22"]
datetime_object=pd.to_datetime(date_list)
data2["date"]=datetime_object # tablo icersine date diye bir alan eklenip içindeki degerler sırasıyla yazılmakta
data2=data2.set_index("date") # sıralama index ini tarih olarak ayarladık
data2
print(data2.loc["2020-03-21"]) # index e göre deger detaylarını inceleme
print(data2.loc["2020-03-20":"2020-03-22"]) # iki index arasındaki degerleri görme
# resample belirlenen degerlere göre işlem yapmak
data2.resample("A").mean() # "M"=month "A"=year yıllara göre ortalama degeri bulmasını istedik

#data2.resample("M").mean() # aya göre ortalamalarını bul
data2.resample("M").first().interpolate("linear") # boş olan degerleri linear artıs seklinde doldur(sadece numerikleri doldurur)
data2.resample("M").mean().interpolate("linear") # mean leri aynı olarak şekilde doldur
data=data.set_index("#") # index degerini 1 den başlatarak indis sırasını düzenledik
data.head()
#data.HP[1]
#data["HP"][1]
data.loc[1,["HP"]] # 1.satır ve HP sütunu kesişimini bul
data[["Attack","HP"]]
print(type(data["HP"]))  #series
print(type(data[["HP"]]))#dataframe

#data["HP"].head()
#data[["HP"]].head()
#data.loc[1:10,"HP":"Defense"]   #1-10 arasında HP ile Defense arasındaki sütunları alma
data.loc[10:1:-1,"HP":"Defense"] #Tersini almak
data.loc[1:10,"Speed":] # Speed den sona kadar al
dataHp= data["HP"]>200
data[dataHp]
first_filter=data.HP>150
second_filter=data.Speed>35
data[first_filter & second_filter]
data.HP[data.Speed<15] # Speed i 100 den büyük olan Pokemonların HP değerini göster
#data.head()

def div(n):
    return n/2
data.HP.apply(div) # HP kolonuna div fonksiyonunu uygula

data.HP.apply(lambda n : n/2) # HP kolonuna lambda içerisindekini uygula
data["TotalPower"]=data.Attack+data.Defense
data.head()
dictionary = {"NAME":["ali","veli","kenan","hilal","ayse","evren"],

              "AGE":[15,16,17,33,45,66],

              "MAAS": [100,150,240,350,110,220]}

dataFrame1 = pd.DataFrame(dictionary)

dataFrame1.iloc[:,2] # 2.kolonun hepsini getir
dataFrame1[dataFrame1.AGE > 60] # yaşı 60 dan büyük kişinin bilgilerini getir
print(data.index.name) # index adını verir
data.index.name="indexName" # Sıralama ya isim verir # yerine yazar
data.head()
data2=data.copy() # data yı kopyalama işlemi
#data2.head()
data2.index = range(100,900,1) # 100 den 900 e kadar 1 er 1 er arttırarak ilerle
data2.head()
data1=data.set_index(["Type 1","Type 2"]) # Type 1 e göre Type 2 değerlerinin gruplandırılması
data1.head(100)
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}
df = pd.DataFrame(dic)
df
# pivoting
df.pivot(index="treatment",columns = "gender",values="response")
df1 = df.set_index(["treatment","gender"])
df1
# lets unstack it
# level determines indexes
df1.unstack(level=0)
# level determines indexes
df1.unstack(level=1)
df2 = df1.swaplevel(0,1) # gender ve treatment ın yerini degiştirmek için
df2
# df.pivot(index="treatment",columns = "gender",values="response")
pd.melt(df,id_vars="treatment",value_vars=["age","response"]) #treatment a göre age ve response tablosu oluştur
# according to treatment take means of other features
df.groupby("treatment").mean()   # mean is aggregation / reduction method # treatment a göre grubla ve ortalamalarını al
# there are other methods like sum, std,max or min
# we can only choose one of the feature
df.groupby("treatment").age.max() #  treatment a göre grupla ve yaslardan max olanı al. max yerine mean ile ortalama da kullanılabilir
# Or we can choose multiple features
df.groupby("treatment")[["age","response"]].min() # treatment a göre grupla age ve response a göre min olanları getir
df.info()
# as you can see gender is object
# However if we use groupby, we can convert it categorical data. 
# Because categorical data uses less memory, speed up operations like groupby
#df["gender"] = df["gender"].astype("category")
#df["treatment"] = df["treatment"].astype("category")
#df.info()
a = {"x":2,"y":3}
b = dict(zip(a.values(),a.keys()))
b
a = [0,1,2,3,4]

for a[0] in a:

    print(a[0])