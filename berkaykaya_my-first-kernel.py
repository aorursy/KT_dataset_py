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
data = pd.read_csv('/kaggle/input/fifa19/data.csv')
data.info() # Tablo Hakkında Bilgiler
data.corr() # Veri Orantısal Karşılaştırmaları
f,ax = plt.subplots(figsize=(18,18))

sns.heatmap(data.corr(),annot = True,linewidth=.5,fmt = '.1f',ax=ax)

plt.show()
data.head(10) # Tablonun baştan 10 tane verisi
data.tail(10) # Tablonun Sondan 10 tane verisi
data.columns # Tablonun Sütunları
# Line Plot

data.Age.plot(kind = 'line',color = 'r',label = 'Age',linewidth = 1,alpha = 0.7,grid = True,linestyle = ':')

data.Stamina.plot(color = 'g',label = 'Stamina',linewidth = 1,alpha = 0.7,grid = True,linestyle = '-.')

plt.legend(loc = 'upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

plt.show()
# Scatter Plot

data.plot(kind = 'scatter',x = 'Overall', y = 'GKReflexes',alpha = 0.5,color = 'blue')

plt.xlabel('Overall')

plt.ylabel('Release Clause')

plt.title('Overall - GKReflexes')
# Histogram

data.ShotPower.plot(kind = 'hist',bins = 50,figsize=(12,12),color='green')

plt.show()
data.ShotPower.plot(kind = 'hist',bins = 50)

plt.clf()
dictionary = {'barcelona':'messi','madrid':'ronaldo','galatasaray':'falcao'}

print(dictionary.keys())

print(dictionary.values())
dictionary['barcelona'] = "Griezman"

print(dictionary)

dictionary['fenerbahce'] = 'Hasanali'

print(dictionary)

del dictionary['barcelona']

print(dictionary)

print('fenerbahce' in dictionary)

dictionary.clear()

print(dictionary) 

# İki kez çalıştırılmıştır.
series = data['Overall']

print(type(series))

data_frame = data[['Overall']]

print(type(data_frame))

x = data['Overall']>90

data[x]
data[np.logical_and(data['Overall']>90,data['Age']>30 )]
data[(data['Overall']>92) & (data['Age']<34)]
lis = [1,2,3,4,5]

for index,value in enumerate(lis):

    print(index,':',value)

print('')



dictionary = {'turkey':'istanbul','usa':'newyork'}

for key,value in dictionary.items():

    print(key,':',value)

print('')



for index,value in data[['Overall']][0:2].iterrows():

    print(index,':',value)