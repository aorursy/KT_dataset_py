# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns # gorsellesty

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/2015.csv')
data.info()
data.corr()
# korelasyon haritasi
f,ax = plt.subplots(figsize = (18,18))
sns.heatmap(data.corr(),annot = True, linewidths = .5, fmt = '.1f', ax=ax)
plt.show()
data.head()
data.columns
#sözlük yaratma, key ve value görüntüleme
sozluk= {'ege': 'denizli', 'karadeniz' : 'trabzon'}
print(sozluk.keys())
print(sozluk.values())
sozluk['ege'] = "izmir"
print(sozluk)
sozluk['marmara'] = "kocaeli"
print(sozluk)
del sozluk['karadeniz']
print(sozluk)
print('ege' in sozluk)
sozluk.clear()
print(sozluk)
series = data['Happiness Score']                   # x = ['asd']    ========> x series oluyor
print(type(series))
data_frame = data[['Happiness Score']]             # x = [['asd']]    ========> x data frame oluyor
print(type(data_frame))
print(series)
print(data_frame)
x=data['Happiness Score']>7
y=data['Happiness Score']<5
data[y]
data[np.logical_and(data['Happiness Score']>7, data['Health (Life Expectancy)']>0.8)]
data[(data['Happiness Score']>7)&(data['Freedom']>0.6)]
i = 5
while i!=0:
        print('i is: ', i)
        i-=1
print(i, 'is equal to 0')
lis = [2,2,3,4,7]
for i in lis:
    print('i is: ', i)
print('')

# index ve value değerlerini çıkarma
for index, value in enumerate(lis):
    print(index," : ",value)
print('')

#sozlukteki key ve value değerlerini öğrenmek için döngüler kullanılabilir.
sozluk = {'ege' : 'denizli', 'karadeniz' : 'trabzon'}
for key,value in sozluk.items():
    print(key," : ",value)
print('')


for index, value in data[['Happiness Score']][2:4].iterrows():
    print(index," : ",value)
def tuble_ex():
    """ return defined t ruble"""
    t=(1,2,3)
    return t
a,b,c = tuble_ex()
print(a,b,c)
print(a,b)
print(a)
x=2
def f():
    x=3
    return x
print(x)
print(f())
x=5    
def f():
    y=2*x       #lokalde x tanımlanmadıysa globaldeki x kullanılır
    return y
print(f())
import builtins
dir(builtins)
# nested function - içiçe fonksiyonlar

def square():
    """karesini döndür"""
    def add():
        x=2
        y=3
        z=x+y
        return z
    return add()**2
print(square())
#default arguments
def f(a,b=2,c=3):
    y = a*b+c
    return y

print(f(5))
print(f(1,2,3))
#flexible arguments *args - değişebilir argümenler - giriş sayısı değişebilir
def f(*deg_arg):
    for i in deg_arg:
        print(i)
f(1)
print("")
f(1,2,3,4)

def f(**kwargs):
    """sozluk key ve value degerlerini ekrana bas"""
    for key,value in kwargs.items():
        print(key," : ",value)
f(bolge = 'ege', sehir = 'denizli', population = 1000000)
#lambda function - hızlı fonksiyon tanımlama
kare = lambda x: x**2
print (kare(4))
kare_kok = lambda x: x**(0.5)
print(kare_kok(4))
toplam = lambda x,y,z: x+y+z
print(toplam(1,2,3))
#anonymous function - lambda function gibi fakat 1 den fazla argümen alabilir

number_list = [2,5,8]
y = map(lambda x:x**3,number_list)
print(list(y))
name = "ronaldinho"
it = iter(name)
print(next(it))
print(*it)
# zip örneği
list1 = [1,2,3,4]
list2 = [5,6,7,8]
z=zip(list1,list2)
print(z)
z_list = list(z)
print(z_list)
un_zip = zip(*z_list)
un_list1,un_list2 = list(un_zip)
print(un_list1)
print(un_list2)
print(type(un_list2))
num1=[5,10,15]
num2=[i**3 if i==10 else i-5 if i<6 else i+5 for i in num1]
print(num2)
#threshold = sum(data.Happiness Score)
ort = sum(data['Happiness Score'])/len(data['Happiness Score'])
print(ort)
data['mutluluk_seviyesi'] = ['mutlu' if i>ort else 'mutlu_degil' for i in data['Happiness Score']]
data.loc[:100,["mutluluk_seviyesi","Happiness Score"]]