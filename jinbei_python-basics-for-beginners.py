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
a=1 #integer



b=1.423   #float



c="314"  #string



str(a) # a değişken stringe dönüştü



float(c) # a değişkeni ondalıklı hale floata dönüştü



int(b)  # b değişkeni integer a dönüştü



round(4.67)  #sayıyı yuvarlar

a=13

b=3



a,b=b,a  #değişkenlerin değerlerini değiştirdi



a,b
b//a  # modunu bulur
b%a   # a'nın b'ye bölümünden kalanı verir
a**2  # üst alma operatörü
c="Hava ne güzel yaa"  #stringleri liste gibi düşünebilirsin



c[0]
c[::-1]  #tersten yazdırır
c[-1]  #sondan da indexler. son karakter -1
c*2  # mat operatörü stringe uygulanırsa o kadar yazdırır
"cimbom".upper()  # hepsini büyütür

lower()

replace("m","n")  # stringteki m karakterleri n ye dönüşür

startswith("ci")  # ci ile başlıyorsa true döndürür

endswith("m")   # m ile bitiyorsa true döndürür

split("")     # verdiğin değere göre ayırır. def= 

strip("")  #verdiğin değere göre strinigin sağından solundan temizler

lstrip()

rstrip()

join("")  #listenin elemanlarının arasına verdiğin karakteri koyar. printteki sep gibi

count("x")  #x'in frekansı

find("x")  # soldan aramaya başlar ilk bulduğunun indedxini verir

rfind()







e="çok"

print("her \nşey {} güzel \tolacak".format(e))



#stringin içindeki süslü paranteze format ile değer verdik.

#\n alt satıra geçti

#\t tab boşluk bıraktı
print(*e)  # * her karakerin arasına boşluk koyar
print(e,"güzel","olacak", sep="'nah'")

#sep parametresi def hali boşluk. değişkenlerin arasına ne gelecek
Liste1=[]

Liste2=list()   #2 şekilde oluşturulabilir



a=[1,2]
a.append(1)

a     # listenin sonuna 1 tane eleman ekler
a.extend([2,3,4]) 

a    # listenin sonuna 1den fazla eleman ekleyebilrsin
a.insert(2,8)

a     # listenin 2. indexine 8i ekledi
a.remove(2)   # listeden elemanı siler. index değil direk elemanı yazıyoruz

a
a.index(1)   # ilk gördüğü 1 kaçıncı indexte onu verir
a.index(1,1)   # 1. indexten sonra aramaya başlar
a.count(1)   # 1'in frekansı
Liste2=[1,2,3,4,5,6]



Liste2.pop(3)   # indeksini belirttiğin elemanı atar. (def=son index)

Liste2
Liste=[3,56,7,2,54,1,22,5,78,9]



Liste.sort()  #küçükten büyüğe sıralar Liste.sort(reverse=True) büyükten küçüğe

Liste
min(Liste)

max(Liste)

sum(Liste)

liste=[1,2,3]



list(map(lambda x : x**2,liste))  # map fonksiyonu liste veya demete(iteratable) fonksiyon uygular

liste=[1,2,3,4]    # 2 listeyi eşleştirir. boyutlarının aynı olmasına dikkat et yoksa eksik veri

liste2=[5,6]

list(zip(liste,liste2))
liste=[1,2,3]

list(enumerate(liste))  # indexleriyle birlikte yazar
list(filter(lambda x: x%2==0, [1,2,3,4,5,6,7,8]))

# fonksiyonu sağlayanlar True yapanlar. yani 2 ye bölümünden 0 olanları döndürdü.
all([True,True,False])  #yalnızca hepsi True ise true döndürür



any([False,False,False])  #yalnızca hepsi false ise false döndürür
demet=(1,1,2,3,4,4,4,4,5,5,6,6,6,6)



demet.count(1)  # 1'in frekansı
demet.index(4)  # ilk 4 değişkeninin indexini verdi
b=dict()   # veritipi dönüşümü ve boş sözlük oluşturdu

a={"bir":1,"iki":2,"üç":3}

a
a["iki"]
a["dört"]=4

a          # sözlüğe eleman ekledik
a.keys()   # anahtarları listeler
a.values()   # değerleri listeler
a.items() # anahtar ve değerleri zipliyor gibi düşün