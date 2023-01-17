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

data = pd.read_csv('../input/world-happiness/2018.csv') # datamızı çekiyoruz
data.info() # data ile ilgili bilgileri görmemizi sağlıyor.
data.corr() # özelliklerin birbiriyle ilişkisini gösteriyor, yani korelasyon .
# korelasyon haritası 

import matplotlib.pyplot as plt

import seaborn as sns

f,ax = plt.subplots (figsize =(10 , 10) )

sns.heatmap (data.corr(), annot = True , linewidths = 3 , fmt = '0.1f', ax = ax)

plt.show()
data.head(6) # ilk 6 ülkenin verilerini bize gösteriyor
data.columns 


# çizgi grafiği

# color = renk, label = etiket, linewidth = çizginin genişliği, alpha = opaklık , grid = ızgara, linestyle = çizgi stili

data.Generosity.plot (kind='line' ,color = 'g' , label = 'Generosity' ,linewidth = 1 , alpha = 0.5 , grid = True , linestyle = '-' )

data.Score.plot (color = 'r' , label ='Score', linewidth = 1 , alpha =0.5 , grid = True , linestyle= '-.')

plt.legend(loc='upper right') 

plt.xlabel("x ekseni")               # label = etiketin ismi

plt.ylabel("y ekseni")

plt.title('çizgi grafiği')

plt.show()

data.plot(kind='scatter', x='Generosity', y='GDP per capita',alpha = 0.5,color = 'red')

plt.xlabel('Generosity')              # label = name of label

plt.ylabel('GDP per capita')

plt.title('GDP per capita ve Generosity Scatter Plot')            # title = başlık

    
data.plot(kind='hist', x='Generosity', y='GDP per capita',alpha = 0.5,color = 'green',grid = True , )

plt.legend(loc='upper right') 

plt.xlabel("Generosity")               # label = etiketin ismi

plt.title('Histogram')

plt.show()
sözlük = {'kitap' : 'içimizdeki şeytan' , 'meyve' : 'elma'}

print(sözlük.keys())

print(sözlük.values())

print(sözlük)
sözlük['kitap'] = 'Kürk mantolu Madonna' # key'in value'sini değiştirebiliriz.

print(sözlük)

sözlük['içecek'] = 'çay'                 # yeni bir key girebiliriz.

print(sözlük)

del sözlük['kitap']                      # istenilen keyi silebiliriz.

print(sözlük)

print('içecek' in sözlük)                # sözlük içinde key araması yapabiliriz. 

sözlük.clear()                           # sözlüğün içindekileri silebiliriz.

print(sözlük)



def topla(a,b,c):

    d =a+b+c

    return d

print(topla(1,2,3,))
def kare():

    def carpim():

        a = 3

        b = 4

        z = a + b

        return z

    return carpim()**2

print(kare())




islem = lambda x : (x+2)**3 

print(islem(3))

toplam = lambda a,b,c : a*b/c

print(toplam(2,3,4))
list1 = [1,2,3]

list2 = ['a','b','c']

print(list2)

z = zip(list1,list2)

zlist = list(z)

print(zlist)
num1 = [1,2,3,4]

num2 = [i**2 if i == 3 else i-5  for i in num1 ]

print(num2)
data.tail()
data.shape
data.describe()
data.boxplot(column = ['GDP per capita'] )
data_yeni = data.head()

data_yeni
melted = pd.melt(frame=data_yeni,id_vars = ['Country or region'] , value_vars='Generosity',)

melted
data1 = data.head()

data2 = data.tail()

data_conc_row = pd.concat([data1,data2],axis = 0, ignore_index = True)

data_conc_row

data1 = data.Generosity.head()

data2 = data.Score.head()

data_conc = pd.concat([data1,data2],axis = 1 )

data_conc
data.dtypes
data.Score = data.Score.astype('int64')
data.dtypes
data.head()
data.info()
data["Perceptions of corruption"].value_counts(dropna = False )
data1 = data

data1["Perceptions of corruption"].dropna(inplace = True)
assert data1["Perceptions of corruption"].notnull().all()
# dictionary'den data frame yapma

Şehir = ["İzmir","Bursa","İstanbul"]  

Nüfus = [4,3,5]

Bölge = ["Ege","Marmara","Marmara"]

list_label = ["Şehir","Nüfus","Bölge"] 

list_col = [Şehir,Nüfus,Bölge] 

zipped= list(zip(list_label,list_col))

data_dict = dict(zipped)

df = pd.DataFrame(data_dict)

df
# Yeni sütun ekleme

df["Ünlü Yapıt"] = ["Saat Kulesi","Ulu Cami","Ayasofya"]

df
# Broadcasting

df["Gelir"] = 0

df
data1 = data.loc[:,["Score","GDP per capita","Social support"]]

data1.plot()
# Subplot

data1.plot(subplots = True)

plt.show()
#Scatter

data1.plot(kind = "scatter", x = "GDP per capita", y = "Social support")

plt.show()
# Hist Plot

data1.plot(kind="hist",y = "Social support" , bins = 50 , range = (0,5) , normed = True )

plt.show()
fig, axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind = "hist",y = "Social support",bins = 50,range= (0,5),normed = True,ax = axes[0])

data1.plot(kind = "hist",y = "Social support",bins = 50,range= (0,5),normed = True,ax = axes[1],cumulative = True)

plt.savefig('graph.png')

plt