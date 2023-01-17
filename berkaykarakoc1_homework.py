# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# .csv uzantımızı  okuttuk ve data.info() kodu ile data bilgilerini gördük

data = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')

data.info()
#data.head kodumuz ise bize ilk 5 datayı getiriyor fakat parantez içerisine data.head(15)

# yazarsak ilk 15 adet datayı getirmiş oluyor

data.head() 
data.corr()
# datalar arasındaki ısı haritası  



f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.columns
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.Global_Sales.plot(kind = 'line', color = 'b',label = 'Global_Sales',linewidth=1,alpha = 1,grid = True,linestyle = ':')

data.Rank.plot(color = 'r',label = 'Rank',linewidth=1, alpha = 1,grid = True,linestyle = '--')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()



# Scatter Plot 

data.plot(kind='scatter', x='Rank', y='Other_Sales',alpha = 0.5,color = 'blue')

plt.xlabel('Rank')              # label = name of label

plt.ylabel('Global_Sales')

plt.title('Attack Defense Scatter Plot') 



y1=data.plot(kind='scatter', x='Year', y='JP_Sales',alpha = 0.5,color = 'blue')

z1=data.plot(kind='scatter', x='Rank', y='Global_Sales',alpha = 0.5,color = 'red')

t1=data.plot(kind='scatter', x='NA_Sales', y='EU_Sales',alpha = 0.5,color = 'red')

plt.title('') 
#Histogram

#bins= kaç adet data olacağını belirtir



x=data.Year.plot(kind='hist',bins=50,figsize=(30,12),color='red')

plt.show(x)
data.columns
# yeni bir sözlük oluşturduk ve anahtar ile değerleri görebiliyoruz

dictionary = {'Turkey':'Trabzonspor','Spain':'Barcelona'}

print(dictionary.keys())

print(dictionary.values())
dictionary['Turkey'] = 'Galatasaray'

print(dictionary)

dictionary['Turkey1'] ='Fenerbahçe'

print(dictionary)

del dictionary['Turkey']

print(dictionary)

print('Spain' in dictionary)

dictionary.clear()

print(dictionary)
data = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')
series = data['Year']    # seri olarak tanımlanır 

print(type(series))

data_frame = data[['Year']]  #dataframe olarak tanımlanır 

print(type(data_frame))

# Karşılaştırma Operatörleri 



print(4 < 5)

print(4 != 4 )



# Boelan Operatörleri

print( True and False )

print( True or False  )
# pandas data frame içerisinden filtreliyoruz



x = data['Year']>2015 

data[x]
# 2 adet datayı filtreliyoruz



data[np.logical_and(data['Year']>2015,data['Genre'] == 'Sports')]
# başka bir 've' operatörü ile çalışma türü iki türlüde kullanılabilir.

data[(data['NA_Sales']>30) & (data['Global_Sales']>25 )]
# 0 dan 10 e kadar sayan program 



i = 0

while i != 10:

    print(' i nin degeri : ', i )

    i += 2 

print(i,'i nin degeri to 10')
# liste üzerinde değer okuması yapma 



lis = [1,2,3,4,5,6,7,8,9,10]

for i in lis:

    print('i is :',i)

print('')



# index ve değerinin karşılaştırılmalı gösterimi 

for index, value in enumerate(lis):

    print(index, ":",value)

print('')



#for yapısını kullanarak oluşturduğumuz sözlüğün içerisinde dolaşmak 

dictionary = {'Turkey':'Trabzonspor','Spain':'Barcelona'}

for key, value in dictionary.items():

    print(key, ":",value)

print('')



# for yapısını kullanarak data tiplerini öğrenmek 



for index,value in data[['Genre']][0:10].iterrows():

    print(index,":",value)