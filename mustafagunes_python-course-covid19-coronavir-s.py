ver1=15

ver2=20

ver3=ver1+ver2

gun="pazartesi" #string

ver4=10.5 #double

print(ver4) 
#%%

ver=10

ver2=20

#%%

ver4=45

ver7=55

#gibi
v="salı"

c=12

d=10.25

veri_tipi1=type(v)

veri_tipi2=type(c)

veri_tipi3=type(d)

print(veri_tipi1)

print(veri_tipi2)

print(veri_tipi3)
ver="123456"

c=len(ver)

print(c)

ver2="trtıotr456t"

c2=len(ver2)

print(c2)

############int tipinde veri girildiğinde çalışmaz####
ver="asderf4567"

print(ver[0])

print(ver[8])

#########aşağıdaki kodda hata verir########

#v=12345

#print(v[0])

########################
print('5'*5)
ver=10.6

c=round(ver)

print(c)

ver2=10.4

c2=round(ver2)

print(c2)
def ilk_fonksiyon(inputlar):

    '''

    parametreler

    

    return:

    

    '''
def fonksiyon(ver1,ver2):

    output=(ver1*ver2)/10

    return output

print(fonksiyon(2,2))

###yada

yazdır=fonksiyon(3,3)

print(yazdır)
def fonksiyon2():

    print("ıkıncı deneme fonksıyon return yazmak şart değil")

fonksiyon2()
def cemberin_cevresi(r):

    '''çemberin çevresini hesaplama input=r output=çevre'''

    output=2*3.14*r

    return output

sonuc=cemberin_cevresi(2)

print(sonuc)
def cemberin_cevresi(r,pi=3.14):

    '''çemberin çevresini hesaplama input=r input=pi sabitlendi output=çevre'''

    output=2*pi*r

    return output

sonuc=cemberin_cevresi(2)

print(sonuc)
#*args kullanımı=fonksiyona sonradan dahil edilecek değişkenleri tutar 

def hesapla(boy,kilo,*args):

    print(args)

    sonuc=boy+kilo

    return sonuc

print (hesapla(15,45,47,4,5))
def hesapla(boy,kilo,*args):

    print(args)

    sonuc=(boy+kilo)*args[0]

    return sonuc

print (hesapla(15,45,47,4,5))
print(round(3.14,2))
yas=10

name="mustafa"

soyad="gunes"

def function(yas,name,*args,ayakkabı_numarasi=35):

    print("cocugun ismi",name,"cocgun yası",yas,"cocgun ayakkabı numarası",ayakkabı_numarasi)

    print(type(name))

    print(float(yas))

    output=args[0]*yas

    return output

sonuc=function(yas,name,soyad)

print("args[0]*yas",sonuc)
def hesapla(x):

    sonuc=x*x

    return sonuc

cıktı=hesapla(3)

print(cıktı)
#lambda ile yazılısı

sonuc=lambda x: x*x

print(sonuc(3))
liste=[1,2,3,4,5]

liste2=["ali",2,"hüseyiin"]

print(liste)

print(liste2)

print(type(liste))

print(type(liste2))

print(liste[2])

print(liste[-1]) #listenin son elamanını verir

print(liste[0:3]) #listenin ilk 3 elamanını verir 0,1,2 indekler

#dir(liste) # liste ile kullanılabilecek metotları verir

liste.remove(5)

print(liste)

liste.append(5)

print(liste)

liste.reverse()

print(liste)
liste=[2,6,5,4,3,9]

liste.sort() # sıralar

print(liste)
dictionary={'ali':35,'memet':45,'ayse':56}

print(dictionary)

print(dictionary['ali'])

print(type(dictionary['ali']))

print(dictionary.keys())

print(dictionary(velues()))
a = "sakin calismaktan vazgecmeyin"

print(a[1]+a[0]+a[8]+a[1] + " " + a[-11:])
ver1=20

ver2=20

if (ver1<ver2):

    print('küçüktür')

elif (ver1==ver2):

    print('esitt')

else:

    print('büyüktür')

    
liste=[1,2,3,4,5]

value=4

if value in liste:

    print('var')

else:

    print('yok')
#girilen yıl değerine göre kaçıncı yüzyıl oldgunu bul

def program(year):

    stringecevir=str(year)

    if (len(stringecevir)<3):

        return 1

    elif(len(stringecevir)==3):

        if ((stringecevir[1:3]) == "00"):

            return int(stringecevir[0])

        else:

            return int(stringecevir[0])+1

    else:

        if (stringecevir[2:4] == "00"):

            return int(stringecevir[:2])

        else:

            return int(stringecevir[:2])+1

           

sonuc=program(1000)

print(sonuc)
for each in range(1,11):

    print(each)

for each in "ankara bolu":

    print(each)
#topalam

liste=[1,2,3,4,5,6,7,8,9,99]

print (sum(liste))

count=0

for each in liste:

    count=count+each

    print(count)
#en küçük sayıyıy bul

liste=[1,2,5,6,66,-999,45,25,36,22,222,565,-89]

mini=10000

for each in liste:

    if (each<mini):

        mini=each

       

    else:

        continue

print(mini)

sonuc=min(liste)

print(sonuc)
# self komutu fonksiyondaki değişkenleri eştlemede kullanılır örnek

class Calisan:

    zam_oranı=1.8

    def __init__(self,isim,soyad,maas):

        self.isim=isim

        self.soyad=soyad

        self.maas=maas

        self.email=isim+soyad+"@"+"gmail"+".com"

    def isinvesoyisim(self):

        return self.isim+"  "+self.soyad

    def zam_yap(self):

        return self.maas+self.zam_oranı*self.maas

isci=Calisan("ali","taş",500)

print(isci.email)

print(isci.maas)

print(isci.isinvesoyisim())

print(isci.zam_yap())
import numpy as np #kullanılacak olan numpy kütüphanesını np kısayoluna atatadık
import numpy as np

array=np.array([1,2,3,4,5,6,7,8,9,10,11,12]) #1x12 vektör oluşturuldu

print(array)

a=array.reshape(3,4) #yukardakı vektörü3x4 vektöre dönüştür

print(a)

print(a.shape) #oluşturlan vektörün boytlatını verir

print(a.ndim) #kaç boyutlu olduğunu gösterir

print(a.size) #vektürün shape edilemden önceki durumu

print(type(a))
#shape komutunu kullnmadan

import numpy as np

array=np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]) #3x4 luk matrıs olustu

print(array)
#sıfırlardan olusan matrıs oollusturma

import numpy as np

a=np.zeros((5,8))

print(a)

#herhangi bir elamnaı güncellemee

a[0,0]=5 # birinci satır birinci sutundakı elamı 5 yap

a[0,5]=15

a[4,7]=15

print(a)
#sadee birlerden olusan matris oluturmaaa,

import numpy as np

a=np.ones((5,6))

print(a)

a[3,3]=15

print(a)
#boş matrıs olusturma

import numpy as np

a=np.empty((3,4))

print(a)
#verlen aralıkta 5 er 5 er artısla rakmları yazdırma

import numpy as np

a=np.arange(5,50,5) # 5 den basla 50 ye kadar 50 dahıl degıl 5 artısla 

print(a)

b=np.arange(1,25,0.85)

print(b)
import numpy as np

a=np.linspace(5,50,20) #5 ten basla 50 ye kadar araya 20 tane sayı yazdır 50 dahıl

print(a)
#toplama çıkarma kare alma sinüs cos

import numpy as np

a=np.array([1,2,3])

b=np.array([4,5,6])

print(a+b)

print(a-b)

print(a**2)

print(np.sin(a))

print(np.cos(b))

print(a<2)
#çarpma

import numpy as np

a=np.array([[1,2,3],[4,5,6],[7,8,9]])

b=np.array([[1,2,3],[4,5,6],[7,8,9]])

print(a*b)

print(a)
#random

import numpy as np

a=np.random.random((5,5)) #0  1 arası olusturur rasgele

b=a.sum()

c=a.max()

print(a)

print(b)

print(c)

print(np.sqrt(a))
#indexing and slicıng

import numpy as np

array=np.array([1,2,3,4,5,6,7])

reverse_array=array[::-1]

print(reverse_array)

array2=np.array([[1,2,3,4,5],[6,7,8,9,10]])

print(array2)

print(array2[1,1])

print(array2[:,1])

print(array2[1,1:4])
#shape manupılatıon

import numpy as np

array=np.array([[1,2,3],[4,5,6],[7,8,9]]) #bunu vektore cevırecek tek boyutlu

a=array.ravel()

print(a)

#tekrar eskı halıne alalım

array2=a.reshape(3,3)

print(array2)

transpozu=array2.T

print(transpozu)

#kaca kaclık matrıs oldugunu ogrenelim

b=transpozu.shape

print(b)
#iki array i birleştirmee

import numpy as np

array=np.array([[1,2],[3,4],[5,6]])

array2=np.array([[-1,-2],[-3,-4],[-5,-6]])

#vertıkal birleştirme dikey

array3=np.vstack((array,array2))

print(array3)

#horizontol birleştirme yatay

array4=np.hstack((array,array2))

print(array4)
#convert and copy listden arraye arrayden listeye dönüşüm

import numpy as np

liste=[1,2,3,4]

array=np.array(liste)

print(array)

liste_geridonus=list(array)

print(liste_geridonus)

print(type(liste_geridonus))

a=np.array([1,2,3])

b=a.copy()

c=a.copy()

b[1]=8

print(a)

print(b)

print(c)
