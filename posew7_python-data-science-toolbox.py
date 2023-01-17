# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/Pokemon.csv")
list = [1,2,3,4,5,6,7]
def tuple_ex():
    t = (1,2,3)
    return t
a,b,c = tuple_ex()
print(a,b,c)

print()

def list_ex():
    l = [1,2,3]
    return l
x,y,z = list_ex()
print(x+1,y+1,z+4)
a = 5
b= 2
def topla():
    b = 6
    return a+b
print(a+b)
print(topla())
import builtins as b
dir(b)
def kare_al():
    def denklem():
        a = 1
        b = 3
        x = a + b
        return x
    return denklem()**2
print(kare_al())
def topla():
    def deger_al():
        a = int(input())
        b = int(input())
        return a+b
    return deger_al()
topla()
def daire_cevre(r, pi = 3.14):
    return 2 * pi * r
daire_cevre(2)

def daire_cevre(r, pi = 3.14):
    return 2 * pi * r
daire_cevre(2, 3)
x = lambda x:x**2
print(x(2))

a = 1
b = 3
c = 4
y = lambda a,b,c:a+b+c
print(y(a,b,c))
list= [1,2,3]
x = [1, 2, 3]
y = [4, 5, 6]
zipped = zip(x, y)
list(zipped)
list1 = [1, 2, 3]
list2 = [4, 5, 6]
zipped = unzip(list1, list2)
list(zipped)
list = [1,2,3,4,5,6,7]

list2 = [i * i for i in list]

list2
list1 = [5,10,15]
list2 = [i**2 if i == 10 else i + 12 if i < 7 else i + i for i in list1]
list2
threshold = sum(data.Speed)/len(data.Speed)
data["speed_level"] = ["low" if i < threshold else "high" for i in data.Speed]
print(threshold)
print(data.loc[:7,["Speed","speed_level"]])

ort_atak = sum(data.Attack)/len(data.Attack)
data["power"] = ["low" if i < ort_atak else "high" for i in data.Attack]
print(ort_atak)
print(data.loc[:7,["Attack","power"]])