# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#verilerin okunması
data = pd.read_csv("../input/heart-disease-uci/heart.csv")
data2 = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv") # missing data için kullanılan db
# veri setinde yeralan future'ların incelenmei
data.columns
# future'ların veri tiplerinin incelenmesi
data.info()
#verinin ön izlemesi
data.head(10)
# future'lar arasındaki korelasyonun incelenmesi
data.corr()
# korelasyonların line plot kullanılarak görselleştirilmesi
data.plot( kind = 'line', x = "target", y = "cp", color = 'g', linewidth = '1', figsize=(12,6) )
plt.xlabel('target')
plt.ylabel('c p')
plt.title('target-cp korelasyon grafigi')
plt.show()

# cinsiyet frekansını histogram ile görselleştirilmesi
data.sex.plot(kind = 'hist',bins = 5, figsize=(12,6))
plt.xlabel('cinsiyet 0:Kadın 1:Erkek')
plt.ylabel('frekans - sayı')
plt.title('Cinsiyet - frekans dağılımı')
plt.show()
# scatter plot kullanımı
plt.figure(figsize=(12,6))
plt.scatter(data.age, data.trestbps, s=30)
plt.scatter(data.age, data.thalach, s=20)
plt.xlabel('yaş')
plt.ylabel('trestbps - thalch')
plt.title('scatter plot')
plt.legend()
plt.show()
yasFiltresi = data["age"] > 65 # 65 yaş üzerindeki hastaların görüntülenmesi
cpFiltresi = data["cp"] > 2 # gögüs agrı tipi 3 olanları filterele

data[yasFiltresi & cpFiltresi] # 65 yas üzeri gögüs agrı tipi 0 ve 1 olan hastaların filtrelenmesi
# 65 yas üzerindeki hasta sayısının öğrenilmesi

hastaSayisi = 0
for each in data.age:
    if each > 65 :
        hastaSayisi += 1
print('65 yaş üzeri hasta sayısı = ',hastaSayisi)
# fonksiyon tanımlama

def getYoungestPatientInfo():
    """ This function returns information who is the youngest patient """
    age = 100;
    
    for each in data.age:
        if each < age:
            age = each
    
    return data[data.age == age]

getYoungestPatientInfo() # tanımlanan fonksiyonun çıktısı

# %% Nested function

def powerRes(i,r):
    """ calculete power of the resistor """
    """ i = current(A) , r = resistor(ohm) """
    
    def square():
        return i**2
    
    return square()*r 

print("power:",powerRes(2,50))  # calculate power for i=2A,r=50ohm
# %% Default and Flexiable function
    
def effectiveValue(Vm,k=0.707):
    """ calculate effective value of the alternative voltage """
    return Vm*k
    

def power(i,v,*cosf):
    """ calculate the power """
    cos = 1
    
    for each in cosf:
        if each != 0:
            cos = each
   
    return i*v*cos


print("Effective value for 380v is : ",effectiveValue(380),"Volt")
print("power for 2A and 220V :",power(2,220), "Watt")
print("power for 2A , 220V and cosfi 0.8 :",power(2,220,0.8), "Watt")
# %% lamda function
    
voltage = lambda i,r:i*r

# calculate voltage for i=2A , r= 50 ohm 
print(voltage(2,50),"Volt")
# %% Anonymous function

res_list = [4,6,2]
result = map(lambda i:12/i,res_list)

print(list(result))
# Iteration
word = "Python"
iterWord = iter(word)
print(next(iterWord))
print(*iterWord)
# Zip

listEng = ["noun","verb","adverb","adjective"]
listTur = ["isim","fiil","zarf","sıfat"]

dictionary = zip(listEng,listTur)
print(dictionary)
wordList = list(dictionary)
print(wordList)
# Unzip
unzip = zip(*wordList)
listEnglish,listTurkish = list(unzip)
print(listEnglish)
print(listTurkish)

# List Comprehension

numbers = [1,2,3,4,5]

squareOfnumbers = [ each**2 for each in numbers]

print("Numbers : ",numbers)
print("Square of the numbers : ",squareOfnumbers)
# Conditionals on iterable

numbers = [1,2,3,4,5]

typeOfNumbers = ["even" if n%2==0 else "odd"  for n in numbers]

print("Numbers are :",numbers)
print("Type of the numbers are : ",typeOfNumbers)

# list comprehension in the dataset

# The dataset includes sex type  1 or 0 therefore we will add new future as male or female.

data["gender"] = [ "male" if each == 1 else "female"  for each in data.sex]

data.loc[:5,["sex","gender"]]
# value counts 
# data içerisinde hangi yaştan kaç hastanın olduğununun value_counts() metodu ile elde edilmesi

print(data.age.value_counts())
# rastegele sayılardan liste oluşturup bu listenin median,lower quantierQ1, upper quantierQ3 incelenmesi

import random 
import math

liste = list()

for i in range(1,12):
    liste.append(random.randint(1,100)) # 1-100 arasında rastgele 11 sayı oluştur

liste.sort() # listeyi sırala
med = math.floor(len(liste)/2)

print("Sayilar : ",liste)
print("Median : ",liste[med]) # medyan 
print("Lower Quantlier Q1 : ", liste[math.floor(med/2)]) # Q1
print("Upper Quantlier Q3 : ", liste[med + math.ceil(((len(liste)-1) - med)/2) ]) #Q3


# veri seti içerisindeki sayısal değerlerin istatiksel özellikleri describe() fonksiyonu ile görüntülenmesi

data.describe()

# boxplot kullanımı : Quantlier değerlerini görselleştirerek analiz yapmayı kolaylaştırır.
data.boxplot(column = 'age')
# melt kullanımı : veri görselleştirmede kolaylık sağlayacak

data2 = data.head()

melted = pd.melt(data2,id_vars = 'age', value_vars = ['cp','chol'])
melted

# concatenating data

# iki tane dataframe'i birleştirmek için kullanılır.

df_1 = data.head()
df_2 = data.tail()

# vertical : dikey yani yukarıdan aşağıya birleştirme axis=0

conc_data_row = pd.concat([df_1,df_2],axis=0,ignore_index=True)


# horizantal : yatay birleştirme

d1 = data["age"].head()
d2 = data["sex"].head()

conc_data_col = pd.concat([d1,d2],axis=1)

conc_data_row

# veri tipini değiştirme

data.dtypes # veri setindeki future'ların veri tipini verir.

# veri seti içerisindeki target future'ı int olarak belirtilmiş 
# ancak bu veri sadece 0-1 değerleri almıştır 
# bu veri tipini int dan bool olarak değiştirdik.

data['target'] = data['target'].astype('bool') # convert int to bool

data.dtypes
# missing data

# dataset içerinde Province/State future'da bulunan missing dataların temizlenmesi 
df = data2["Province/State"].dropna(inplace=True)

assert data2["Province/State"].notnull().all() # yaptığımız işlemi kontrol etme. eğer başarılıysa hata vermeyecek