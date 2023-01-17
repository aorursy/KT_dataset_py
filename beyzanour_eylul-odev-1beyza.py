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
import pandas as pd

import matplotlib.pyplot as plt



df1 = pd.read_excel("py_mind.xlsx", 'emrullah', ignore_index=True)

py_mind = ['osman', 'ozgur', 'emrah', 'riza', 'zeyd', 'ufuk', 'Mehmet A', 'Murat Z', 'Hakan']

py_sense = ['Ramazan', 'Faruk', 'Huseyin', 'Adnan', 'Kaya', 'Zafer', 'Mathchi', 'Fikret Y', 'Emrah']

py_science= ['Zerin','beyza','Esra','Sehri','Emine S.','Emine T.', 'Rukiye']

py_opinion=['Emrah U','Salih','Murat O','Aydin','Ibrahim','Ensar','Mehmet K','Naim']

siniflar = [py_mind, py_sense, py_science, py_opinion]

isimler = ['py_mind', 'py_sense', 'py_science', 'py_opinion']

#df_merge=pd.concat(df1,df2, axis = 0)

for j in range(4):

    for i in siniflar[j]:

        name = isimler[j] + '.xlsx'

        df2= pd.read_excel(name, i, ignore_index=True)

        df1=df1.assign(x=df2["ogr.C"])

        #print(df1.columns)

        name = df2.columns[0]

        df1 = df1.rename(columns={'x': name})



#for i in py_opinion:

 #   name = 'py_opinion' '.xlsx'

  #  df2= pd.read_excel(name, i)

   # df1=df1.assign(x=df2["ogr.C"])

   # name = df2.columns[0]

   # df1 = df1.rename(columns={'x': name})

        

print(df1)

print(len(df1.columns)-2)#giren kisi sayisi

print(df1.loc[df1['Emrullah Gulcan']=='Dogru'])

print(df1.iloc[[20, 21, 22]]) #dogru yanlis bos sayilari

df7 = df1.iloc[[20]].drop(['ogr.C', 'Emrullah Gulcan'], axis=1 )

df7.values[0].sort()

print(df7.values[0][-4:-1]) #en cok dogru yapan 3 kisi

print(df7.values[0][0:3]) # en az dogru yapan kisi



for j in range(4):

    maxi = 0

    best = ''

    for i in siniflar[j]:

        name = isimler[j] + '.xlsx'

        df2= pd.read_excel(name, i, ignore_index=True)

        dogru = df2['ogr.C'].iloc[[20]].values[0]

        if(dogru>maxi):

            maxi = dogru

            best = i

    print(isimler[j], best, maxi) # heer sinifta en basarili kisi, selcuk silindi

dogrulistesi=[]

yanlislistesi=[]

for j in range(20):

    dogru=0

    yanlis=0

    

    a=df1['Cevap A.'].iloc[[j]].values[0]

    

    for k in df1.columns[3:]:

        b=df1[k].iloc[[j]].values[0]

        if a == b:

            dogru+=1

        elif a!= b:

            yanlis+=1

    print(j+1, dogru)#her sorunun dogru yapilma miktari

    print(j+1, yanlis)#her sorunun yanlis yapilma miktari 

    dogrulistesi.append(dogru)

    yanlislistesi.append(yanlis)

print(dogrulistesi.index(max(dogrulistesi))+1, max(dogrulistesi))

print(yanlislistesi.index(max(yanlislistesi))+1, max(yanlislistesi))

liste=range(1,21)

df=pd.DataFrame({"Id":liste,"dogru":dogrulistesi,"yanlis":yanlislistesi})

print(df)

plt.plot(df.Id, df.dogru, color="red", label="dogru")

plt.plot(df.Id, df.yanlis, color="blue", label="yanlis")

plt.legend()

plt.xlabel("Id")

plt.ylabel("Dogru Sayisi")

plt.show()

plt.scatter(df.Id, df.dogru, color="red", label="dogru")

plt.scatter(df.Id, df.yanlis, color="blue", label="yanlis")

plt.show()

plt.hist(df.dogru, bins=5)

plt.xlabel("Dogrular")

plt.ylabel("frekans")

plt.title("hist")

plt.show()
