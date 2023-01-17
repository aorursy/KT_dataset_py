# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/pokemon-challenge/pokemon.csv')
data.info()
#correlation map(2 özellik(feature) arasındaki değer 1 ise bunlar birbiri ile doğru orantılıdır)

#correlation: ilişki

f,ax = plt.subplots(figsize=(13, 13))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt=".1f",ax=ax)

plt.show() #yazıyı siler
data.corr()
data.head(10)
data.columns
data.Speed.plot(kind ='line', color='g',label='Speed',linewidth=1, alpha=0.5, grid=True, linestyle=':')

data.Defense.plot(color='r', label='Defense', linewidth=1, alpha=0.5, grid=True, linestyle='-.')

plt.legend(loc='upper right')

plt.xlabel('x axis') #x çizgimizin adı

plt.ylabel('y axis') #y çizgimizin adı

plt.title('Line Plot') #başlık

plt.show()
data.plot(kind='scatter', x='Attack', y='Defense', alpha=0.5,color='red')

plt.xlabel('Attack')

plt.ylabel('Defence') 

plt.title('Attack Scatter Plot')    

#plt.show() :alttaki siyah satırı kaldırmaya yarar
data.Speed.plot(kind='hist', bins=50, figsize=(15,15))

plt.show()
#clf() :çizdirdiğimiz plotu temizler 

data.Speed.plot(kind='hist', bins=50)

plt.clf()
dictionary ={'spain': 'madrid' , 'usa' : 'vegas'}

print(dictionary.keys())

print(dictionary.values())
dictionary['spain'] = "barcelona" #güncelleme yapıyor

print(dictionary)

dictionary['france'] = "paris"

print(dictionary)

del dictionary ['spain'] #silme işlemi yapar

print(dictionary)

print ('france' in dictionary) #fransa kütüphanenin içinde mi -->true döndürdü

dictionary.clear() #tamamen siler

print(dictionary)
#del dictionary --> böyle bırakırsak yukarda sildiğimiz için hata verir

print(dictionary) 
data = pd.read_csv('../input/pokemon-challenge/pokemon.csv')
series = data['Defense']

print(type(series))

data_frame = data[['Defense']]

print(type(data_frame))
#comparison(karsılastırma) operator

print(3>2)

print(3!=2)



#boolean operator

print(True and False)

print(True or False)
#Filtering 

x=data['Defense']> 200

data[x]  #defansı 200 den büyük olan pokemonları yazdırır
data[np.logical_and(data['Defense']>200, data['Attack']>100)] #defansı 200den, atağı 100den büyük pokemonlar
data[(data['Defense']>200) & (data['Attack']>100)] #üstteki kod ile aynı işlemi görür
i = 0

while i != 5:

    print('i is: ', i)

    i +=1

print (i, 'i equal to 5')
lis = [1,2,3,4,5] #liste oluşturduk

for i in lis: #i listenin içinde dolaşıyor

    print('i is: ',i)

print('') 





for index, value in enumerate(lis): #index değerleri ile listeyi gösterir

    print(index, " : ", value)

print('')





dictionary = {'spain':'madrid' , 'france':'paris'}

for key,value in dictionary.items():

    print(key," : ", value) #kısaca anahtar ve değerlere ulaşırız

print('')





for index, value in data[['Attack']][0:1].iterrows(): #atağın değeri ve indeksini bulur

    print(index, " : ", value)
def tuble_ex():

    """return defined t tuble"""

    t = (1,2,3)

    return t

a,b,c = tuble_ex()

print(a,b,c)
x = 2

def f():

    x = 3 

    return x

print(x)

print(f())
x = 5

def f():

    y = 2*x

    return y 

print(f())
import builtins

dir(builtins)
def square():

    """return square of value"""

    def add():

        """add two local variable"""

        x = 2 

        y = 3

        z = x+y

        return z 

    return add()**2

print(square())
def f(a,b = 1, c=2):

    y = a + b + c

    return y

print(f(5))



print(f(5,4,3))
def f(*args):

    for i in args:

        print(i)

f(1)

print("")

f(1,2,3,4)



def f (**kwargs):

    """print key and value of dictionary"""

    for key, value in kwargs.items():

        print(key, " " , value)

f(country = 'spain', capital='madrid', population= 123456)
#long way

def square (x):

    return x**2

print(square(5))



#short way

square = lambda x: x**2

print(square(4))

tot = lambda x,y,z: x+y+z

print(tot(1,2,3))
number_list = [1,2,3]

y = map(lambda x: x**2, number_list)

print(list(y))
name = "ronaldo"

it = iter(name)

print(next(it))

print(*it)
list1 = [1,2,3,4]

list2 = [5,6,7,8]

z = zip(list1,list2)

print(z)

z_list = list(z)

print(z_list)
un_zip = zip(*z_list)

un_list1,un_list2 = list(un_zip)

print(un_list1)  #un zip yaparken köşeli parantez kullanılmaz

print(un_list2)

print(type(un_list2))



print(type(list(un_list1))) #unlisti listeye cevir type' ına bak
num1 = [1,2,3]

num2 = [i + 1 for i in num1 ] #listedeki verileri bana +1 ekleyip return et

print(num2)
num1 = [5,10,15]

num2 = [i**2 if i == 10 else i-5 if i<7 else i+5 for i in num1]

print(num2)
threshold = sum(data.Speed)/len (data.Speed)

print("threshold",threshold)

data["speed_level"] =  ["high" if i > threshold else "low" for i in data.Speed]

data.loc[:10, ["speed_level","Speed"]]
data = pd.read_csv('../input/pokemon-challenge/pokemon.csv')

data.head() #ilk 5 satırı verir
data.tail() #son 5 satırı verir
data.columns 
data.shape #kaç değer olduğunu verir
data.info() #data ile ilgili bilgiler verir
print(data['Type 1'].value_counts(dropna=False))
data.describe() #sadece numeric değerleri verir
data.boxplot(column='Attack', by='Legendary')

plt.show()
data_new = data.head() #ilk 5 pokemonu gösterir

data_new
melted = pd.melt(frame= data_new, id_vars='Name', value_vars=['Attack','Defense'])

melted
melted.pivot(index='Name', columns='variable', values='value')
data1 = data.head() #ilk 5 pokemon

data2 = data.tail() #son 5 pokemon

conc_data_row = pd.concat([data1,data2],axis=0, ignore_index=True)

conc_data_row
data1 = data['Attack'].head()

data2 = data['Defense'].head()

conc_data_col = pd.concat([data1,data2],axis=1)

conc_data_col
data.dtypes
data['Type 1'] = data['Type 1'].astype('category') #data type'ın tipini category olarak değiştirir

data['Speed'] = data['Speed'].astype('float') #speed'in tipini float olarak değiştirir
data.dtypes
data.info()
data["Type 2"].value_counts(dropna = False) #type2 de her farklı değerden kaç tane var bunu hesapla diyor

#dropna = False --> null değer varsa eğer onu da göster
data1 = data

data1["Type 2"].dropna(inplace = True)

#type 2 nin null olanları listeden at 

#inplace = True --> çıkar ve çıkardığın sonuçları data1 in içine kaydet
assert 1 == 1 #yukardaki kodun işe yarayıp yaramadığını gösterir

#hiçbir sey döndürmediği için bu doğru demektir
assert data['Type 2'].notnull().all() #hepsi doğru mu diyor(boş olanları listeden attık)

#doğru olduğuu için bir sey döndürmez
data["Type 2"].fillna('empty', inplace = True) #type2 yi empty ile doldurur
assert data['Type 2'].notnull().all()
assert data.columns[1] == 'Name' #ilk sütunun adı 'name' mi --> doğru olduğu için geriyye değer döndürmez
data.Speed.dtypes
assert data.Speed.dtypes == np.float
#dataframe oluşturma

country = ["Spain","France"]

populatıon = ["11","12"]

list_label = ["country","populatıon"]

list_col = [country,populatıon]

zipped = list(zip(list_label,list_col))

data_dict = dict(zipped)

df = pd.DataFrame(data_dict)

df
#add new column(yeni sütun ekleme)

df["capital"] = ["madrid","paris"] #capital sütunu oluşturup 2 yeni değer oluşturdu

df
df ["income"] = 0 #yeni sürun oluşturup hepsine 0 değerinş atadı

df
data1 = data.loc[:,["Attack","Defense","Speed"]]

data1.plot()
#subplots

data1.plot(subplots = True)

plt.show()
#scatter plot

data1.plot(kind="scatter", x="Attack", y="Defense")

plt.show()
data1.plot(kind = "hist", y="Defense", bins = 50, range= (0,250))
fig, axes = plt.subplots(nrows= 2, ncols=1)

data1.plot(kind="hist", y="Defense", bins=50, range=(0,250),  ax=axes[0])

data1.plot(kind="hist", y="Defense", bins=50, range=(0,250),  ax=axes[1], cumulative= True)

plt.savefig('graph.png')

plt
time_list = ["1992-03-08","1992-04-12"]

print(type(time_list[1]))



datetime_object = pd.to_datetime(time_list)

print(type(datetime_object)) 
import warnings

warnings.filterwarnings("ignore")



data2 = data.head()

date_list = ["1992-01-10", "1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datetime_object = pd.to_datetime(date_list)

data2["date"]  =datetime_object



data2 = data2.set_index("date")

data2
print(data2.loc["1993-03-16"])

print(data2.loc["1992-02-10":"1993-03-16"])
data2.resample("A").mean() #yıla göre resample et ve ortalamalarını al
data2.resample("M").mean() #aylara göre resmaple et
data = pd.read_csv('../input/pokemon-challenge/pokemon.csv')

data = data.set_index("#") #datamızın indeksini # yap

data.head()
data["HP"][1]
data.HP[1]
data.loc[1,["HP"]]
data[["HP","Attack"]]
print(type(data["HP"]))

print(type(data[["HP"]]))
data.loc[1:10, "HP":"Defense"] #1 den 10a kadar olan pokemonların can ve defansını al
data.loc[10:1: -1, "HP":"Defense"] #tersten alma
data.loc[1:10,"Speed": ] #en sonuncusana kadar al
boolean = data.HP >200 #datanın canı 200den büyük olanlar

data[boolean] #sadece true olanları yazdırır
first_filter = data.HP>150

second_filter = data.Speed > 35

data[first_filter & second_filter]
data.HP[data.Speed<15] #hızının 15ten küçük olduğu pokemonların canı
def div(n):

    return n/2 #canları yarıya böler

data.HP.apply(div) 
data.HP.apply(lambda n: n/2)  #def in kısa hali
data["total_power"] = data.Attack + data.Defense #diğer columları kullanarak yeni colum oluşturabilirz

data.head()
print(data.index.name)  #indexlerin adını değiştirdik



data.index.name= "index_name" #index_name yaptık

data.head()
data = pd.read_csv('../input/pokemon-challenge/pokemon.csv')

data.head()
data1 = data.set_index(["Type 1","Type 2"])

data1.head(100)
dic = {"tedavi":["A","A","B","B"], "cinsiyet":["F","M","F","M"],"tedavicevap":[10,45,5,9],"yaş":[15,4,72,65]}

df = pd.DataFrame(dic)

df
df.pivot(index ="tedavi", columns ="cinsiyet", values="tedavicevap")
df1= df.set_index(["tedavi","cinsiyet"])

df1
df1.unstack(level=0)
df1.unstack(level=1)
df2 = df1.swaplevel(0,1)  #indekslerin yerini değiştirebiliriz

df2
df
pd.melt(df,id_vars="tedavi",value_vars=["yaş","tedavicevap"]) #tedavi sabit kalır
df
df.groupby("tedavi").mean() #tedaviye göre grupla ortalamasını al
df.groupby("tedavi").yaş.max()  #max yaşı bulur 
df.groupby("tedavi")[["yaş","tedavicevap"]].min()