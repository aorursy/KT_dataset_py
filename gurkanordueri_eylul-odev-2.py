# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from  openpyxl import *

from xlrd import open_workbook





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os





# Any results you write to the current directory are saved as output.
mind_exel = open_workbook("/kaggle/input/examresult/py_mind.xlsx")

opinion_exel = open_workbook("/kaggle/input/examresult/py_opinion.xlsx")

science_exel = open_workbook("/kaggle/input/examresult/py_science.xlsx")

sense_exel = open_workbook("/kaggle/input/examresult/py_sense.xlsx")







#Sınıflardaki öğrenciler

mind_ogrenciler = mind_exel.sheet_names()

opinion_ogrenciler = opinion_exel.sheet_names()

science_ogrenciler = science_exel.sheet_names()

sense_ogrenciler = sense_exel.sheet_names()





#Sınıflardaki öğrencilerin sayisi

mind_uyeleri = len(mind_ogrenciler)-1

opinion_uyeleri = len(opinion_ogrenciler)-1

science_uyeleri = len(science_ogrenciler)-1

sense_uyeleri = len(sense_ogrenciler)-1







mind_bolum = 0

opinion_bolum = 0

science_bolum = 0

sense_bolum = 0

#Data Frame olusturuldu

tum_siniflar = pd.DataFrame(columns=("SINIF", "ISIM", "DOGRU", "YANLIS", "BOS", "DYB"), index=None)






#Siniflarin exel dosyalari icerisinde gezinerek ogrenci listeleri yazdirilacak; dogru, yanlis ve bos sonuclari cikartilacak; siniflar arasi karsilastirma yapilacak hale getirilecek

# bunlar olusturdugumuz data frame e yazdirilacak.





while mind_bolum < mind_uyeleri:

    mind_bolum += 1 # her tabloda dolasabilmek icin sayac.

    mind_liste = pd.read_excel("/kaggle/input/examresult/py_mind.xlsx", sheet_name=mind_bolum)

    mind_sonuc = mind_liste.tail(3) #Dogru yanlis ve boslar alindi

    DYB = list(mind_liste.iloc[0:20,1]) # Ogrencilerin seceneklerini sectik

    sinif_ismi = "py_mind"

    isim = mind_sonuc.columns[0] # 0 indexli satir ve sutunu alir bu bizim ogrenci ismimiz olacak

    isim = str(isim)

    dogru = mind_sonuc.iloc[0,1] # Dogru sayisini

    yanlis = mind_sonuc.iloc[1,1] # yanlis sayisini

    bos = mind_sonuc.iloc[2,1] # bosu alir

    ogrenci = {'SINIF': sinif_ismi, 'ISIM': isim, 'DOGRU': dogru, 'YANLIS': yanlis, "BOS": bos, "DYB":DYB}    

    tum_siniflar = tum_siniflar.append(ogrenci, ignore_index=True) # duzenli olan dic'i,  olusturdugumuz bos DF ye ekledik. her bir ogrenci ve sinif icin tum bunlar yapilacak

    

while opinion_bolum < opinion_uyeleri:

    opinion_bolum += 1

    opinion_liste = pd.read_excel("/kaggle/input/examresult/py_opinion.xlsx", sheet_name=opinion_bolum)

    opinion_sonuc = opinion_liste.tail(3)

    DYB = list(opinion_liste.iloc[0:20,1])

    sinif_ismi= "py_opinion"

    isim = opinion_sonuc.columns[0]

    isim = str(isim)

    dogru = opinion_sonuc.iloc[0,1]

    yanlis = opinion_sonuc.iloc[1,1]

    bos = opinion_sonuc.iloc[2,1]

    ogrenci = {'SINIF': sinif_ismi, 'ISIM': isim, 'DOGRU': dogru, 'YANLIS': yanlis, "BOS": bos, "DYB":DYB}    

    tum_siniflar = tum_siniflar.append(ogrenci, ignore_index=True)





while sense_bolum < sense_uyeleri:

   

    sense_bolum += 1

    sense_liste = pd.read_excel("/kaggle/input/examresult/py_sense.xlsx", sheet_name=sense_bolum)

    sense_sonuc = sense_liste.tail(3)

    DYB = list(sense_liste.iloc[0:20,1])

    sinif_ismi = "py_sense"

    isim = sense_sonuc.columns[0]

    isim = str(isim)

    dogru = sense_sonuc.iloc[0,1]

    yanlis = sense_sonuc.iloc[1,1]

    bos = sense_sonuc.iloc[2,1]

    ogrenci = {'SINIF': sinif_ismi, 'ISIM': isim, 'DOGRU': dogru, 'YANLIS': yanlis, "BOS": bos, "DYB":DYB}    

    tum_siniflar = tum_siniflar.append(ogrenci, ignore_index=True)





"""while science_bolum < science_uyeleri:

    science_bolum += 1

    science_liste = pd.read_excel("/kaggle/input/examresult/py_science.xlsx", sheet_name=science_bolum)

    science_sonuc = science_liste.tail(3)

    DYB = list(science_liste.iloc[0:20,1])

    sinif_ismi = "py_science"

    isim = science_sonuc.columns[0]

    isim = str(isim)

    dogru = science_sonuc.iloc[0,1]

    yanlis = science_sonuc.iloc[1,1]

    bos = science_sonuc.iloc[2,1]

    ogrenci = {'SINIF': sinif_ismi, 'ISIM': isim, 'DOGRU': dogru, 'YANLIS': yanlis, "BOS": bos, "DYB":DYB}    

    tum_siniflar = tum_siniflar.append(ogrenci, ignore_index=True)"""



    









    

print(tum_siniflar)


#siniflarin ayri ayri df yapilmasi ve ortalama alinmasi

c_mind = tum_siniflar[tum_siniflar["SINIF"]=="py_mind"]

c_mind_ort = c_mind["True"]

c_sense = tum_siniflar[tum_siniflar["SINIF"]=="py_sense"]

c_sense_ort = c_sense["True"]

c_science = tum_siniflar[tum_siniflar["SINIF"]=="py_science"]

c_science_ort = c_science["True"]

c_opinion = tum_siniflar[tum_siniflar["SINIF"]=="py_opinion"]

c_opinion_ort = c_opinion["True"]

print("sinif ortalamasi","\npy_mind ortalamasi =", c_mind_ort.mean(), "\npy_Sense ortalamasi =", c_sense_ort.mean(),

      "\npy_science ortalamasi =",c_science_ort.mean(), "\npy_opinion ortalamasi =", c_opinion_ort.mean())
#sinif sinif en basarili kisiler

mind_1st= c_mind.sort_values(by='True', ascending=False).head(1)

sense_1st= c_sense.sort_values(by='True', ascending=False).head(1)

science_1st= c_science.sort_values(by='True', ascending=False).head(1)

opinion_1st= c_opinion.sort_values(by='True', ascending=False).head(1)

print("py Mind en basarili ogrenci =", mind_1st.Name, "\npy Sense en basarili ogrenci =", sense_1st.Name, "\npy Science en basarili ogrenci =", science_1st.Name,"\npy Opinion en basarili ogrenci =", opinion_1st.Name)
#siniflarin standart sapmalari

mind_std = print(c_mind["True"].std())

sense_std = print(c_sense["True"].std())

science_std = print(c_science["True"].std())

opinion_std = print(c_opinion["True"].std())
# Grafiklestirme modulu

import matplotlib.pyplot as plt

#sinif ortalamalarini

y = np.array([c_mind_ort.mean(),c_sense_ort.mean(),c_science_ort.mean(),c_opinion_ort.mean()])

x = np.array([1,2,3,4])



#bar plot

ind = np.arange(5)

plt.bar(x,y)

plt.title("bar plot")

plt.xlabel("x cubugu")

plt.ylabel("y cubugu")

plt.xticks(ind, (" ", 'mind', 'sense', 'science', 'opinion'))

plt.show()