# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataFrame = pd.read_csv("../input/2017.csv")
dataFrame.info()
dataFrame.columns
dataFrame.corr()
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(dataFrame.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()

print(dataFrame.head(10))
print(dataFrame.tail(10))
dataFrame.rename(columns={'Happiness.Score': 'Happiness_Score','Happiness.Rank': 'Happiness_Rank',
                          'Whisker.high': 'Whisker_high','Economy..GDP.per.Capita.':'Economy_GDP_per_Capita','Whisker.low': 'Whisker_low',
                          'Trust..Government.Corruption.':'Trust_Government_Corruption',
                           'Dystopia.Residual': 'Dystopia_Residual', 'Health..Life.Expectancy.': 'Health_Life_Expectancy'}, inplace=True)

dataFrame.columns
dataFrame.describe()
filtered_dataFrame= dataFrame.Happiness_Rank<dataFrame.Happiness_Rank.mean() 
flt_dF=dataFrame[filtered_dataFrame]
print(flt_dF)

x = dataFrame['Happiness_Rank']<78     # 
dataFrame[x]
# 2 - Filtering pandas with logical_and
 # logical_and numpy kütüphanesine bağlı bir keyword
dataFrame[np.logical_and(dataFrame['Happiness_Rank']<20, dataFrame['Happiness_Score']>7 )]
plt.scatter(dataFrame.Happiness_Score,dataFrame.Health_Life_Expectancy,alpha = 0.5,color="orange",label="dataFrame")

plt.legend()
plt.xlabel("Happiness_Score")
plt.ylabel("Health_Life_Expectancy")
plt.title("scatter plot")
plt.show()
plt.plot(dataFrame.Happiness_Score,dataFrame.Health_Life_Expectancy,color="blue",label="virginica")
plt.xlabel('Happiness_Score')
plt.ylabel('Health_Life_Expectancy')
plt.legend()
#dataFrame.plot(grid=True, alpha=3)  #▲alpha çizginin saydamlığını gösteriyor
plt.show()
dataFrame.Happiness_Score.plot(kind = 'hist',bins = 80
                               ,figsize = (10,10))

plt.show()
#Dicrionary
#creating dictionary and look its keys and values

#listelerden daha hızlı oldukları için  dictionary tercih ediyoruz
dictionary = {'Fatih' : 31 ,"Ahmet" : 32,(1,2,3):"a,b,c"}
print(dictionary.keys())
print(dictionary.values())

# Keys have to be immutable objects like string, boolean, float, integer or tubles
# List is not immutable
# Keys are unique
dictionary['Fatih'] = "Adana"    # update existing entry
print(dictionary)
dictionary['Hasan'] = "Adıyaman"       # Add new entry
print(dictionary)
del dictionary['Fatih']              # remove entry with key 'spain'
print(dictionary)
print('Hasan' in dictionary)        # check include or not  True olarak dönüyor burası
#dictionary.clear() #remove all entries
print(dictionary)
#TUBLE
tuble_example = ('fatih', 100 , 3.14, 'Ahmet', 49.9 )
#tuble_example[1]=25 değiştiremiyoruz TypeError: 'tuple' object does not support item assignment
print(tuble_example)
 # ilgili indisteki elemanı yazdırabiliyoruz ama değiştiremiyoruz
def tbl_examplex():
    """ return defined t tuble"""   #  for documentation of function
    t = (1,2,3)
    return t
a,b,c = tbl_examplex()
print(a,b,c)



dataFrame.columns
#while basit döngüsü, kural tamamlanana kadar işi yapar durur.
i = 0
while i != 5 :      # i eşit değil 5 olana kadar i yi arttır ve bunu print et
    print('i is: ',i)
    i +=1 
print(i,' is equal to 5') 

#list içerisinde döngü kontrolü



# Stay in loop if condition( i is not equal 5) is true
lis = [1,2,3,4,5,6,7,8]
for i in lis:       #for döngüsü ile listenin içine girip kontroller yapıyoruz/yapacağız
    print('i is: ',i)
    

# Enumerate index and value of list     
#index : value = 0:1, 1:2, 2:3, 3:4, 4:5
for index, value in enumerate(lis):   # listenin hem indexine hem value sine erşimek için 
                            #her öğeye erişmek için for döngüsünü kullanıyoruz ayrıca enumerate metodunu da kullanırız.
    print(index," : ",value)
    
#for ile dictionarty öğelerine de erişebiliyoruz
for key,value in dictionary.items():
    print(key," : ",value) 
# pandas ta for ile içeriğe erişebiliyoruz 
for index,value in dataFrame[['Happiness_Score']][1:2].iterrows():      #iterrows metodu ilgili kordinatta[0:1] daki value yu bulup onun indexinide buluyor
    print(index," : ",value)

x = 2  # --global değişken
def f():
    x = 3   # ---- fonsiyonun içinde tanımlanmış değişken
    return x
print(x)      # x = 2 global scope
print(f())    # x = 3 local scope

#local scope yok ise globali kullanır
z=7
def trt():
    q=z**2
    return q
print(trt())
#built in scopes
# dir ile içeriği çağırabiliyoruz
import builtins
dir(builtins)
#nested functions

def square_root():
    def ilk_topla():
        a=3
        b=6
        c=a+b
        return math.sqrt(c)
    return ilk_topla()
print(square_root())
# default arguments
def f(x, y = 3, z = 6):   # y ve z ye default değerler verilmiştir
    t = x + y ** z
    return t
print(f(5))   # fonksiyona yazılan değer inputlardan boş olan a ya aittir
# fonksiyona diğer inputları ezecek değişkenler verilebilir
print(f(5,4,3))
# flexible arguments *args,* kwargs
def f(*args):
    for i in args:
        print(i)
f(1)
f(4,5,6,8) # print dahil metoddur.
# **kwargs  dictionary ler için  flexible arguments dir
def f(**kwargs):
    """ dictionary için key ve value ler yazılır."""
    for key, value in kwargs.items(): 
        print(key, " ", value)
f(fatih=21,hasan=22,ömer=23,hüseyin="henüz reşit değil")
#lambda funtion
def square_rooty(x):     # sesli olarak yazmak istersek- square_rooty metodu x inputu alıyor  
    return math.sqrt(x)   #  x in kare kökünü return ediyor. sonra print içerisine metodu yazıp
print(square_rooty(25))    #inputu veriyoruz

# lambda function
square_root_lambda = lambda x: math.sqrt(x)     # where x is name of argument
print(square_root_lambda(36))

tot = lambda x,y,z: x+y+z   # where x,y,z are names of arguments
print(tot(1,2,3))

# ANONYMOUS FUNCTİON
sayi_listesi=[25,36,49,64,81]
y=map(lambda x: math.sqrt(x),sayi_listesi)
print(list(y))

name="FATİH"  #name variable, iter fonksiyon, it obje, next it ilk harfini yazdırıyor.
it= iter(name)
print(next(it))
# liste zipleme 
list1 = [1,2,3,4]
list2 = [5,6,7,8]
z = zip(list1,list2)    # z bir zip objesi oldu , bunu listeye çevirmek için list metodunu kullanıyoruz
print(z)
z_list = list(z)
print(z_list)
#unzip 
un_zip = zip(*z_list)   #oluşan un_zip bir obje listeye çevirmek için list metodu..
un_list1,un_list2 = list(un_zip) # unzip returns tuble
print(un_list1)
print(un_list2)
print(type(un_list2))
print(type(list(un_list1)))
type(z)
# list comprehension
num1 = [1,2,3]           #itarable obje(num1)yi itaretor(for) ile itartion yapıyoruz..
num2 = [i + 1 for i in num1 ]   #list comprehension
print(num2)
# Conditionals on iterable
num1 = [5,10,15]
num2 = [i**2 if i == 10 else i-5 if i < 7 else i+5 for i in num1]
#kodu anlatırken yazmaya şuradan başlıyoruz dedi:
# for i in num1 kodu buradan başlayarak anlayacağım
 # bir döngü var, bu döngü sırasında i yi koşullamış  1) if i == 10 (i 10 ise i**2),
   #  else i-5 if i < 7 burayı şöyle okuyoruz elseif i<7 ise i-5,
#else i+5 kısmı ise 10 a eşit değil ve 7 den küçül değil ise demek i+5
# sonuç [0, 100, 20]
print(num2)
dataFrame.columns
threshold = sum(dataFrame.Happiness_Score)/len(dataFrame.Happiness_Score)
print("threshold",threshold)
dataFrame["Happiness_Level"] = ["high" if i > threshold else "low" for i in dataFrame.Happiness_Score]
dataFrame.loc[:10,["Happiness_Level","Happiness_Score"]] # we will learn loc more detailed later








