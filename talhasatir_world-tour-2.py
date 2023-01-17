# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data =pd.read_csv('../input/heart-disease-uci/heart.csv')
#Verimizin ilk 10 satırına bakıyoruz =ne ne deyil diye

data.head(10)
#Data hakkında bilgi alıyorum

data.info() 
#data kolon başlıklarını aldım

data.columns
lis=[8,4,5,9,7,35]

for index,i in enumerate(lis):

    print(index,':',i)

#tuble =çok verileri dizide deyilde fonksiyonda tutma yöntemi

def tubl_ex():

    t=(1,2,'t')

    return t

a,b,c =tubl_ex() #a=1,b=2,c=t degerlerini,karekterini tutar

print((a+b)*c)
#iç içe fonksiyon

def ilk(x,y):

    def son(x):

        return x**2

    return son(x)*y

ilk(5,4)

#default fonksiyonun kullanımı

def f(a,b=3,c=4):

    return a*b*c

print(f(1))

print(f(1,2))#b defouldunu degistirdim c yine defould degerde

print(f(1,4,2))#b,c defouldunu degistirdim
#args =boyutsuz liste gibi paremetrede kaç deger atarsam o kadar deger almıs oluyor

def c(*args):      #NOT:degisken isminin illa args olmasına gerek yok.

    for i in args:

        print(i) 



c(2)

print('uzayli')

c(2,4,5)

#kwargs =distinory i  gösterme

def s(**kwargs):

    for key,value in kwargs.items():

        print(key,':',value)

        

s(country ='Turkiye',capital='istanbul',population=456654)



#lambda fonk.= fonksiyonu hızlı bir şekilde olusturmaya yarar

square = lambda x:x**2 #gelen sayının karesini alma

print(square(4))



extraction =lambda x=1,y=2,z=3: 10-x-y+z #islem defult degerler ile

print(extraction())

#list comprehension

num1 =[1,2,3]

num2 =[i+1 for i in num1] #dizi elemanlarının 1 fazlasını num2 ye at

print(num2)
num1 =[5,10,15]

num2 =['ali' if i>10 else 'mehmet' if i>5 else 'kenan' for i in num1]

print(num2)