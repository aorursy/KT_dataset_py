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
# read csv file

df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

# show top 5 files

df.head()
# show colums's name of dataframe

df.columns
# this method shows colums type int, float or boolean etc. 

df.info()
# Correlation map 

# 2 özellik arasındaki ilişliyi gösterir 

df.corr().head()
f,ax = plt.subplots(figsize=(14,14))

sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
# grafiğin boyutunu ayarlama

plt.figure(figsize=(10,6))



# Grafikte görülecek datalar

df.trestbps.plot(kind= 'line', color='purple', label='Trestbps', alpha=.5, linestyle='-')

df.chol.plot(kind='line', color='b', label='Chol', linestyle=':')



plt.legend(loc='upper right')

plt.xlabel('x axis') #kişi sayısını gösteriyor diye düşünüyorum

plt.ylabel('y axis')

plt.title('Trestbps - Chol Line Plot')

plt.show()
f = plt.figure(figsize=(12,6))



#grafik koordinatlarını belirleme

ax1 = f.add_axes([0.1, 0.1, 0.9, 0.9])

ax2 = f.add_axes([0.6, 0.68,0.3,0.3])



#verileri grafiğe ekleme

ax1.plot(df.chol, color='pink')

ax2.plot(df.age, color='purple')



# X ekseni başıklar

ax1.set_xlabel('Chol')

ax2.set_xlabel('# of person')



# y ekseni başlıklar

ax1.set_ylabel('Chol')

ax2.set_ylabel('Age')
df.plot(kind='scatter', x='trestbps', y='thalach', color='orange', alpha='0.4')

plt.xlabel('trestbps')

plt.ylabel('thalach')

plt.title('trestbps - thalach')
df.age.plot(kind='hist', color='orange', bins=35, figsize=(7,5), grid=True)

plt.show()



# bu grafikte kişilerin en çok 58-59 yaşlarında oldukları verisine varılabilinir.
dataframe = df['age']>60

df[dataframe].head()
# Yaşı 40'dan küçük, trestbps'i 120 olan kişiler

dataframe = df[ (df['age']<40) & (df['trestbps']==120) ]

dataframe
# dictionary'de for loop

dataframe = df['age']

count1=0

count2=0

count3=0

count4=0



for key, value in dataframe.items():

    if value<30:

        count1+=1

    elif value>=30 and value<40:

        count2+=1

    elif value>=40 and value<50:

        count3+=1

    else:

        count4+=1

        

print(count1,' kişi 30dan küçüktür')

print('')

print(count2,' kişi 30 ile 40 yaş arasındadır')

print('')

print(count3,' kişi 40 ile 50 yaş arasındadır')

print('')

print(count3,' kişinin yaşı 50den büyüktür')