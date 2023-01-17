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
# tuble



def tuble_ex():

    t = (1,2,3)

    return t

a,b,c = tuble_ex()

print(a,b,c)
# scope



x = 2 # global, main body

def f():

    x = 3 # local, fonk. içinde

    return x



print(x)  # X=2
print(f()) # x=2
x = 5

def f():

    y = 2*x

    return y

print(f())   # no local, so uses global
# built-in

import builtins

dir(builtins)
# nested function

def square():

    def add():

        x = 2

        y = 3

        z = x + y

        return z

    return add()**2



print(square())
# default arguments:



def f(a, b=1, c=2):

    y = a+b+c

    return y

print(f(5))



# 5+1+2
# flexible args-1

def f(*args):

    for i in args:

        print(i)

        

f(1)
# flexible args-2,  for lists

def f(*args):

    for i in args:

        print(i)

        

f(1,2,3,4)
# flexible args-3, for dictionary (kwargs)

def f(**kwargs):

    for k, v in kwargs.items():

        print("key is:", k, ", values is: ", v)

        

f(country="spain", capital="madrid", population=1234)
# long way - write a fuction:

#karesini alan fonk.

def square(x):

    return x**2

print(square(5))
# short way- lambda func:



square = lambda x: x**2

print(square(4))
tot = lambda x,y,z: x+y+z

print(tot(1,2,3))
# for lists:



mlist = [1,2,3]

y = map(lambda x: x**2, mlist)

print(list(y))
#example

name = "samuel"

it = iter(name)

print(*it)
#example2

name2 = "samuel"

it2 = iter(name2)

print(next(it2))

#remaining:
#remaining:

print(*it2)
# zip

lista = [1,2,3]

listb = [4,5,6]



#merge:

z = zip(lista, listb) # z objesi yaratıldı. listeye çevir:

z_list = list(z)

print(z_list)
# unzip:

un_z = zip(*z_list) #un_z objedir, listeye çevir:

un_lista,un_listb = list(un_z)  #returns tuple



print(un_lista)

print(un_listb)
print(type(un_listb)) #tuple dir. Lİsteye çevirmek gerekir:

print(type(list(un_listb)))
# list, comprehension



n1 = [1,2,3]  #list, iterable object

n2 = [i+1 for i in n1]  #i+1: list comprehension; for i in n1: for loop



# 1+1, 2+1, 3+1

print(n2)

# conditional, iterabşe



n1 = [5,10,15]

#n1 elemanlarına bak(for i in n1), 

#değer=10 ise karesini al, 

#değer<7 ise değer-5, 

#harici ise değer+5

# 5-5, 10*10, 15+5

n2 = [i**2 if i==10 else i-5 if i<7 else i+5 for i in n1]

print(n2)
# pokempon database ine geri dönelim

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

#ilk 10 kaydı inceleyelim, genel bakış:

datapok = pd.read_csv("../input/pokemon-challenge/pokemon.csv")

datapok.head(4)
# Speed leri, ort.Speed(threshold) e göre sınıflandır,low veya high

# ort = toplam / sayı

thold = sum(datapok.Speed) / len(datapok.Speed)

print(thold)
# yeni kolon(feature) açıp, sınıfını doldur:

#datapok Speed elemanlarına bak: for i in datapok.Speed

# değer>ort ise high, harici ise low

# if i>thold high, else high



datapok["speed_level"] = ["high" if i>thold else "low" for i in datapok.Speed]



#istediğin kolonları listele, ilk 8 kayıt gelsin:

datapok.loc[:8, ["speed_level", "Speed"]]