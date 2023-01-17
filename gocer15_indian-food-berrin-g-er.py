# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

data = pd.read_csv('/kaggle/input/indian-food-101/indian_food.csv')
data.info()
data.corr()

#data.corr(feature'lar arasındaki baağlantıyı verir -1/0/1 gibi)

##hocam, buradaki 1 oranını nasıl yorumlamam gerekiyor!!!!
#correlationmap hazırlamak için:

fig, ax = plt.subplots(figsize = (11,11))

sns.heatmap(data.corr(), annot = True, linewidths = .5, fmt='.2f', ax = ax)

plt.show()
data.columns
data.prep_time.plot(kind = 'line', color = 'g',label = 'prep_time',linewidth=1.5,alpha = 1,grid = True,linestyle = ':')

data.cook_time.plot(kind = 'line', color = 'red',label = 'cook_time',linewidth=1.5,alpha = 1,grid = True,linestyle = 'solid')

plt.legend(loc='upper right')     

plt.xlabel('x axis')              

plt.title('Line Plot')            

plt.show()
data.cook_time.plot(kind = 'line', color = 'red',label = 'cook_time',linewidth=1.5,alpha = 1,grid = True,linestyle = 'solid')

plt.show()
data.prep_time.plot(kind = 'line', color = 'g',label = 'prep_time',linewidth=1.5,alpha = 1,grid = True,linestyle = ':')

plt.show()
data.plot(kind = 'scatter', x = 'cook_time',y = 'prep_time',linewidth=1.5,alpha = 0.5,color='red',figsize=(8,8))

plt.show()
#iki data arasında bağlantı yok diyebiliriz.. ?

#hocam yorumlama kısmı ile ilgili önerebileceğiniz kitap video vs var mı acaba?
data.cook_time.plot(kind = 'hist',bins = 50,figsize = (7,7))

plt.show()
data.prep_time.plot(kind = 'hist',bins = 50,figsize = (7,7))

plt.show()
data_frame = data[['region']]

print(data_frame)

#hocam burada '...' olan kısımları neden göstermiyor aslında farklılar!!!?
data.head(10)
data.columns
data[(data['prep_time']>20) & (data['cook_time']>40)]
data[(data['region'] == 'East') | (data['region'] == 'West')]
data[(data['diet'] == 'vegetarian') & (data['course'] == 'dessert')]

#burada sayı olarak 255 çeşit var ve bunlar şunlardır şeklinde liste haline nasıl dökebilirim acaba?!!!
for index,value in data[['region']][9:10].iterrows():

    

    print(index," : ",value)
for index,value in data[['region']].iterrows():

    

    print(index," : ",value)
for index,value in data[['diet']][0:1].iterrows():

    print(index,":",value)

#burada belli bir aralığı almak için ne yapmam gerek acaba ?!!!