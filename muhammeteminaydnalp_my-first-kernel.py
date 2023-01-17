# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data1 = pd.read_csv("/kaggle/input/youtube-new/FRvideos.csv")

print(data1)

data1.info()
data1.head()
data1.corr()
f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data1.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data1.columns
# Line Plot



data1.likes.plot(kind = 'line', color = 'g',label = 'likes',linewidth=1,alpha = 0.9,grid = True,linestyle = ':')

data1.views.plot(kind = "line", color = 'r',label = 'views',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend()     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot 



# x = dislikes, y = views

data1.plot(kind='scatter', x='dislikes', y='views',alpha = 0.7,color = 'blue')

plt.xlabel('dislikes')              # label = name of label

plt.ylabel('views')

plt.title('dislikes views Scatter Plot')
# Scatter Plot 



# x = likes, y = views

data1.plot(kind='scatter', x='likes', y='views',alpha = 0.7,color = 'blue')

plt.xlabel('likes')              # label = name of label

plt.ylabel('views')

plt.title('likes views Scatter Plot')
# Histogram

# bins = number of bar in figure

data1.plot(kind = 'hist',bins = 50,figsize = (12,12))



plt.clf()
dictionary = {"meyve" : "elma", "sebze" : "pirasa" }

print(dictionary)

print(dictionary.keys())

print(dictionary.values())



dictionary["meyve"] = "armut"

print(dictionary)

dictionary["et"] = "balık eti"

print(dictionary)

del dictionary["meyve"]

print(dictionary)

print("sebze" in dictionary)

dictionary.clear()

print(dictionary)





print(dictionary)

data2 = pd.read_csv("/kaggle/input/youtube-new/USvideos.csv")

data2.head()

data2.info()
# 1 - Filtering Pandas data frame



filtre = data2["likes"]>5000000

#data2[filtre]

filtre2 = data2["views"]>100000000

data2[filtre & filtre2]
data2[np.logical_and(data2['dislikes']>500000, data2['likes']>1000000 )]

i = 0

while i != 5 :

    print('i is: ',i)

    i +=1 

print(i,' is equal to 5')
lis = [1,2,3,4,5]

for i in lis:

    print('i is: ',i)

print('')
for index, value in enumerate(lis):

    print(index," : ",value)

print('') 



dictionary = {'spain':'madrid','france':'paris'}

for key,value in dictionary.items():

    print(key," : ",value)

print('')



for index,value in data1[['likes']][0:3].iterrows():

    print(index," : ",value)