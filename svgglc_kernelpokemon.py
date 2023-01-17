# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/combats.csv')

data.info()
data.corr() #bir evin oda sayısı artarsa fiyatta artar
#correlation map
f,ax = plt.subplots(figsize=(15, 15))
#18 128 boyut
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
#annot sayı yazsınmı linewitdh karler arası bosluk fmt:sıfrdan sonra kaç basamak
plt.show()
data.head(10)
data.columns

data.First_pokemon.plot(kind='line',color='r',label='firstp',linewidth=0.07,alpha=0.5,grid=True,linestyle=':')
data.Second_pokemon.plot(kind='line',color='g',label='sec',linewidth=0.07,alpha=0.5,grid=True,linestyle=':')
data.Winner.plot(kind='line',color='y',label='win',linewidth=0.07,alpha=0.5,grid=True,linestyle=':')
plt.show()
data.First_pokemon.plot(kind='line',color='r',label='firstp',linewidth=0.007,alpha=0.5,grid=True,linestyle=':')
data.Second_pokemon.plot(kind='line',color='g',label='sec',linewidth=0.07,alpha=0.5,grid=True,linestyle=':')
data.Winner.plot(kind='line',color='y',label='win',linewidth=0.07,alpha=0.5,grid=True,linestyle='-.')
plt.legend(loc='upper right')
plt.xlabel('pokemonlar')
plt.ylabel('veriable')
plt.title('pokeman grafigi')
plt.show()
data.plot(kind='scatter',x="Winner",y='First_pokemon',alpha=0.05,color='red')
plt.xlabel('kazanan')
plt.ylabel('birincipokemon')
plt.title("grafigim")
#histogram
data.Winner.plot(kind='hist',bins=400,grid=True,figsize=(20,20))
plt.show()
#winner 200 olanın sayısı 700
#bins cubuk sayısı
dic={'spain':'madrid','usa':"vegas"}
print(dic.keys())
print(dic.values())
dic['spain']="bercolona"
dic['france']='paris'
del dic['spain'] #silme key
print(dic)
serieler=data['First_pokemon']
data_frameim=[('First_pokemon')]
#filtreleme
x=data['First_pokemon'] >500

#and or
data[np.logical_and(data['First_pokemon'] >800, data['Second_pokemon']>700)]
i=0
while i != 5 :
    print("i sayısı",i)
    i+=1
print("bitti")
listem=[1,2,3,4]
for i in listem:
  print("deger:",i)
print('bitti')

#index ve degeri
for indisimiz ,deger in enumerate (listem):
    print(indisimiz,"0>>",deger)
print('')
dicti={'x':'madrid','y':"vegas"}
for keyimiz,deger in dicti.items():
    print(keyimiz,'----',deger)
print('')
for indis,deger in data[['First_pokemon']][0:5].iterrows():
    print(indis,'---',deger)
print('')
#ilk 5ini al