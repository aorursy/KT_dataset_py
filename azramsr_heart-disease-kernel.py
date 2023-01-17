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
df=pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

df.head()

#database tanımladım ve ilk 5 veriyi yazdırdım.
df.info()

#database içerisindeki değişkenler
f,ax = plt.subplots(figsize=(14, 14))

sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()

#birbirleriyle olan ilişkilerini gösterdim.0 üstündeyse aralarındaki ilişki pozitif(doğrusal) değilse negatif(etkisiz).
# Line Plot

df.age.plot(kind = 'line', color = 'b',label = 'Age',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

df.trestbps.plot(color = 'r',label = 'Trestbps',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     

plt.xlabel('x axis')              

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()

#Age ve trestbps oran 0.3 bu orana göre ilişkisini gösterdim.
# Scatter Plot 

df.plot(kind='scatter', x='age', y='trestbps',alpha = 0.5,color = 'red')

plt.xlabel('Age')              # label = name of label

plt.ylabel('trestbps')

plt.title('Age trestbps Scatter Plot')  

# Histogram

df.trestbps.plot(kind = 'hist',bins = 50,figsize = (10,10))

plt.show()

#trestbps histogram şeklindeki değerleri
dictionary = {'age' : 'trestbps','thalach' : 'chol'}

print(dictionary.keys())

print(dictionary.values())
#PANDAS



x = df['trestbps']>120     

df[x]

#kanbasıncı 120 den yüksek kişiler
#kan basıncı 120 den yüksek ve yaşı 70 den büyük

df[np.logical_and(df['trestbps']>120, df['age']>70 )]
dictionary = {'age' : 'trestbps','thalach' : 'chol'}

for key,value in dictionary.items():

    print(key," : ",value)

print('')
for index,value in df[['trestbps']][0:1].iterrows():

    print(index," : ",value)