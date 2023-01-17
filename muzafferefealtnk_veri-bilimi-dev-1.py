# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #Datamızdaki belirli verileri grafik işlemlerine çevirerek görselleştirme yapıyoruz

import seaborn as sns # visualization için kullandıgımız kütüphane görselleştirmeyi sağlar







import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/bitcoin-historical-data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv")

data.info()
data.corr()
f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.head(10) #Tablonun ilk on değeri 
data.columns#Veri seti içindeki tablo sütünları
data.corr()


# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.High.plot(kind = 'line', color = 'g',label = 'High',linewidth=1,alpha = 0.5,grid = True,linestyle = '-')

data.Low.plot(color = 'r',label = 'Low',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # plt.show kaldırırsa plotun türünu yazar plt.title

plt.show()
data.columns

# Scatter Plot 

# x = attack, y = defense

data.plot(kind='scatter', x='High', y='Low',alpha = 0.5,color = 'red')

plt.xlabel('High')              # label = name of label

plt.ylabel('Low')

plt.show()# title = title of plot
#Matplotlib kısayol tablo çizimi

plt.scatter(data.High,data.Low)
data.head(1)
# bins = number of bar in figure

data.High.plot(kind = 'hist',bins = 50,figsize = (12,12),color='red')

plt.show()
data.High.plot(kind = 'hist',bins = 50)

#plt.clf()#plottan sonra bunu kullanırsan plotu ortadan kaldırır.
data = pd.read_csv('../input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv')
data.head(5)
series = data['High']        # data['Defense'] = series

print(type(series))

data_frame = data[['High']]  # data[['Defense']] = data frame

print(type(data_frame))
print(3 > 2) #3 büyük oldugu içib 1 den retrun ifadesi olarak mantıksal olarak 1 degeri gelir.

print(3!=2) #3 eşit olmadıgı için 2 ye mantıksal olarak 1 döner

# Boolean operators

print(True and False) #and 1 tane 0 varsa hep mantıksal olarak 0 değeri döner

print(True or False) #or 1 tane 1 olması durumunda mantıksal olarak 1 değeri döner
# 1 - Filtering Pandas data frame

x = data['High']>200     # False değerlerini göstermez 200 den büyük olanları yani True olanlar gelir.

data[x]
# 2 - Filtering pandas with logical_and

# İki şartı da aynı anda sağlayan değerler döner.

data[np.logical_and(data['High']>200, data['Low']>100 )]
# & ve işareti yukarıda anlattıgım ile aynı (and)

data[(data['Defense']>200) & (data['Attack']>100)]
# i 0 i degeri 5 e eşit oldugu an while döngüsünden çıkar ve 5 değerine eşit şeklinde mesaj döndürür.

i = 0

while i != 5 :

    print('i is: ',i)

    i +=1 

print(i,' is equal to 5')
#-----------------------------------------------

# liste eleman sayısı kadar for döngüsü 5 defa tekrarlar her seferinden 1 is 1, 2 is 2 yazar

lis = [1,2,3,4,5]

for i in lis:

    print('i is: ',i)

print('')

#-----------------------------------------------

# Enumerate index and value of list

# index : value = 0:1, 1:2, 2:3, 3:4, 4:5 

#Hangi değerin kaçıncı index de oldugunu bize söyler.

for index, value in enumerate(lis):

    print(index," : ",value)

print('')   

#-----------------------------------------------------

# For dictionaries

# İndex le aynı mantık sadece string değer var.

dictionary = {'spain':'madrid','france':'paris'}

for key,value in dictionary.items():

    print(key," : ",value)

print('')

#-----------------------------------------------------

# For pandas we can achieve index and value

for index,value in data[['High']][0:1].iterrows():

    print(index," : ",value)