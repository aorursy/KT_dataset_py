# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sb
import math
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/2017.csv")
data.columns
data.info()
newdata=data.sample(20)
newdata
data.corr()
f,ax=plt.subplots(figsize=(18,18))
sb.heatmap(data.corr(),annot=True,linewidth=.5, fmt=".1f",ax=ax)
plt.show()
data["Happiness.Score"].plot(color="red",figsize=(12,12))
plt.legend()
plt.xlabel("id")
data["Family"].plot(kind="hist",bins=50,label="Family",color="orange",figsize=(12,12))
plt.legend()
plt.show()
data.plot(kind="scatter",x="Health..Life.Expectancy.",y="Happiness.Score")
plt.show()
newdata.plot(kind="bar",x="Economy..GDP.per.Capita.",y="Trust..Government.Corruption.",color="darkblue",figsize=(12,12))
plt.show()
def Raise(x):
    return math.ceil(x)
data["Freedom"]=data["Freedom"].apply(Raise)
print(data["Freedom"])
FilteredData=data[(data["Economy..GDP.per.Capita."]>1.5) & (data["Happiness.Score"]>5.5)]
FilteredData
data["Happiness"]=["low" if each<4 else  "average"  if each<6 else "high" for each in data["Happiness.Score"]]
data
data.columns
def func(x):
      def func1(y):
            return y+5
      if x>0:
          return func1(x)
      elif x==0:
          return x
      else:
          return func1((-x))
sayi=-5
sonuc=func(sayi)
sonuc
def funclist(*args):
    for each in args:
        print(each)
        
def funcdict(**dicti):
    for key,value in dicti.items():
        print(key,"  ",value)
liste=[12,35,62,89]
funclist(liste)
funcdict(Yas=[24,52,38],City=["LA","PARİS","İSTANBUL"])
    
    
sayi =lambda x:x**2
sayi(9)
numbers1=[2,4,6,8,10]
numbers2=map(lambda x:x*2,numbers1)
print(list(numbers2))

value1=[20,30,40,50,60,70]
value2=[each if each<45 else each+10 if each<60 else each+each  for each in value1]
value2
def defaultfunction(a,x=10,y=14):
    return a+x**2+y
defaultfunction(5)
defaultfunction(10,10,10)
liste1=[4,3,2,1]
liste2=[8,7,6,5]
liste3=zip(liste1,liste2)
zip_liste3=list(liste3)
zip_liste3
un_zip=zip(*zip_liste3)
unzip_liste1,unzip_liste2=list(un_zip)
print(unzip_liste1)
print(unzip_liste2)