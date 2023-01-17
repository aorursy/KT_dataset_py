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
#User Defined Function



def Tanı():

    İsimler="Ali","Veli","Kenan"

    return İsimler

A,V,K=Tanı()

print(A,V,K) 



def Tanı2():

    İsimler2="Ali","Veli","Kenan"

    return İsimler2

A,_,K=Tanı2() # Eğer Eşleştirilecek bir değer yok ise Alttire(_) işareti konulabilir

print(A,K)
#Scope

#Yerel Ve Global Değişkenler Arasındaki Farkı Belirtir



pi=3.14                   #Global Değişken

def AlanSilindir():

    r=6                   #Yerel Değişken

    return 4*pi*r**2

print(AlanSilindir())



def CevreDaire():

    r=23                  #Yerel Değişken

    return 2*pi*r

print(CevreDaire())
#Nested Function

# İç İçe Fonksiyonları Kapsar



def YazDök():

    Maas=2285

    def ZamYap():

        if(Maas<2500):

            return Maas*0.15+Maas

        else:

            return Maas

    return "Verilecek Olan Maaş ",ZamYap()*0.09+Maas



print(YazDök())
#Default Args

#Bazı Değerlerin Bir İşlemde Sabit Kaldığını Belirtmek İçin Kullanılır



def SilAlan(r,h,pi=3.14):

    return pi*r**2*h

print(SilAlan(0.30,48))

print("______")



#Flexible Args

# "*args" Sayesinde Bir İşlemde Sınırsız Sayıda Sayı Gönderilebilir



def HesTop(*args):

    top=0

    for i in args:

        top=top+i

    print(top)

HesTop(6,8,9,1,67)

print("________")



#Flexible Args2

# "Kwargs" Sayesinde Bir Listeleme Ve Eşleştirme İşlemi Yapabiliriz



def Eslestir(**kwargs):

    for key,value in kwargs.items():

        print(key,value)

Eslestir(Country="= Turkey",Capital="= Ankara",Population="= 84.000.000")



#Lambda Function

#Methodların Hızlı Olarak Yazılabilmesini Sağlar

pi=3,14

DaireAlan=lambda r,h,pi=3.14:pi*r**2*h

DaireAlan(5,50)



#Anonymous Function

#Lambda Fonksiyon Gibir Fakat Bir Listede Bulunan Tüm Değerler İçin Sırasıyla Hesap Yapabilir



SilindirYcaplar=[0.250,0.500,0.750,1]

SilindirHacim=map(lambda r,pi=3.14,h=48:pi*r**2*h,SilindirYcaplar)

list(SilindirHacim)



#Zip Methodu

#İki Listenin İndexlerine  Göre Birleşmesini Sağlar

List1=["İstanbul","Van","Mersin","Sinop"]

List2=[34,65,33,57]

Birlestir=zip(List1,List2)

Birlestirildi=list(Birlestir)

print(Birlestirildi)



#Zip Bozma

#Yukarıdaki Örnekği Eski Haline Çevirir



Birlestirboz=zip(*Birlestirildi)

Listee1,Listee2=list(Birlestirboz)

print(Listee1)

print(Listee2)


#List Comprehension

Num1=[10,100,1000]

Num2=[i*(i+1)/2 for i in Num1]

print(Num2)



Dictionary={"Ülkeler":["Türkiye","Almanya","Japonya"],

            "Başkentler":["Ankara","Berlin","Tokyo"],

            "Nüfus_Yaş_Ort":[32,35,40]}

          

Data=pd.DataFrame(Dictionary)

Ortyaş=Data.Nüfus_Yaş_Ort.mean() 

Data["Nüfus Durumu"]=[ "Kötü" if i>Ortyaş else "İyi" for i in Data.Nüfus_Yaş_Ort]

print(Data.Nüfus_Yaş_Ort.mean() )