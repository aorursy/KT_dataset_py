# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv('../input/fifa19/data.csv')
data.info()
data.corr()
f,ax = plt.subplots(figsize=(50, 50))

sns.heatmap(data.corr(), annot=True, linewidths=.8, fmt= '.1f',ax=ax)

plt.show()
data.head(15) #Listedeki en iyi 15 futbolcuyu göster.(Show the greatest 15 players on the list.)
data.columns #sütunlar (features)
data.Finishing.plot(kind='line',color='r',label='Finishing',linewidth=1,alpha=0.5,grid=True,linestyle=':')

data.Penalties.plot(color = 'b',label ='Penalties',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot') 

plt.show()# title = title of plot
data.plot(kind='scatter', x='FKAccuracy', y='LongShots',alpha = 0.5,color = 'b')

plt.xlabel('FK Accuracy')              # label = name of label

plt.ylabel('Long Shots')

plt.title('FK Accuracy-Long Shots Scatter Plot')            # title = title of plot
data.SprintSpeed.plot(kind = 'hist',bins = 50,figsize = (30,30))

plt.show()
data.head()
dictionary={"L.Messi":94,"Cristiano Ronaldo":94,"Neymar Jr":92,"De Gea":91,"K.De Bruyne":91}

dictionary
data.tail() #Listenin sonunu görelim.

dictionary["G.Nugent"]=46

dictionary
dictionary.keys() #Sözlüğün anahtarları yani oyuncu isimleri
dictionary.values() #Sözlükteki değerler yani oyuncu overalları
dictionary.items() #tüm sözlük
print("L.Messi" in dictionary)
print('E.Hazard' in dictionary)
for key,value in dictionary.items():

     print("Name:",key,"\nOverall:",value)

print("")
dictionary.clear() #Sözlüğü temizleyelim

dictionary 
data.head()
p_age=data["Age"]>41 #Yaşı 41'den büyük oyuncu sayısını bulalım.

data[p_age]
data[np.logical_and(data['Overall']>85, data['Age']<20 )] #Overalli 85 ten fazla ve 20 yaşından küçük oyuncu listesi

data[(data['Overall']>85) & (data['Age']>36)] #Bu da bir başka filtering gösterimi burada da 40 yaşından büyük ve overalli 85 ten fazla olan oyuncuları inceledik.