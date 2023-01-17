import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_excel("../input/quranfull.xlsx")

df.head()
bol=19
sozlukA={'ا':1,'إ':1,'أ':1,'آ':1,'ء':1,'ب':2,'ت':400,'ث':500,'ج':3,'ح':8,'خ':600,'د':4,'ذ':700,'ر':200,'ز':7,'س':60,'ش':300,'ص':90,'ض':800,'ط':9,'ظ':900,'ع':70,'غ':1000,'ف':80,'ق':100,'ك':20,'ل':30,'م':40,'ن':50,'ه':5,'هـ':5,'ة':5,'و':6,'ؤ':6,'ئ':10,'ى':10,'ي':10}
liste2=[]

ekle=[]

for ayet in df.ayet.values:

    #vektörleştir

    ic=[]

    liste=[]

    ic2=[]

    

    for i in ayet:

        if(i==" "):

            liste.append(ic)

            ic=[]

        else:

            ic.append(sozlukA[i])

    liste.append(ic)

    liste



    #19 kontrol

    kontrol1=[]

    kontrol2=[]

    kontrol3=[]

    kontrol4=[]

    kontrol5=[]

    kontrol6=[]



    for i in range(len(liste)):

        kontrol1.append(i+1)

        kontrol1.append(len(liste[i]))

    kontrol1b=int("".join([str(i) for i in kontrol1]))

    ic2.append(kontrol1b%bol)

    

    kontrol2 = kontrol1.copy() 

    for i in range(3, 2*len(liste), 2):

        kontrol2[i] = kontrol2[i-2]+kontrol1[i]

    kontrol2b=int("".join([str(i) for i in kontrol2]))

    ic2.append(kontrol2b%bol)



    for i in range(len(liste)):

        kontrol3.append(i+1)

        kontrol3.append(sum(liste[i]))

    kontrol3b=int("".join([str(i) for i in kontrol3]))

    ic2.append(kontrol3b%bol)



    kontrol4 = kontrol3.copy() 

    for i in range(3, 2*len(liste), 2):

        kontrol4[i] = kontrol4[i-2]+kontrol3[i]

    kontrol4b=int("".join([str(i) for i in kontrol4]))

    ic2.append(kontrol4b%bol)



    for i in range(len(liste)):

        kontrol5.append(i+1)

        kontrol5.extend(liste[i])

    kontrol5b=int("".join([str(i) for i in kontrol5]))

    ic2.append(kontrol5b%bol)



    kontrol6 = kontrol5.copy() 

    for i in range(len(liste[0])-1):#ilk kutu farklı

        kontrol6[i+2] = kontrol6[i+1]+kontrol5[i+2]

    toplam=0

    for i in range(1,len(liste)):#liste 1,2,3

        toplam+=len(liste[i-1])+2

        for j in range(len(liste[i])):

            #print(i," ",j," ",toplam+j)

            if(j==0):

                kontrol6[toplam+j] = kontrol6[toplam+j-2]+kontrol5[toplam+j]

            else:

                kontrol6[toplam+j] = kontrol6[toplam+j-1]+kontrol5[toplam+j]

        toplam-=1

    kontrol6b=int("".join([str(i) for i in kontrol6]))

    ic2.append(kontrol6b%bol)

    liste2.append("-".join([str(i) for i in ic2]))

    

    tot=(kontrol1b%bol)+(kontrol2b%bol)+(kontrol3b%bol)+(kontrol4b%bol)+(kontrol5b%bol)+(kontrol6b%bol)

    if(tot==0):

        print("fits the system ",ayet)

    ekle.append(tot)
df["19"]=liste2

df