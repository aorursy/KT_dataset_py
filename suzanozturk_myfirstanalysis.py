# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/winemag-data_first150k.csv')
data.info()

data.dtypes  #all columns types

mean_point = data.points.mean()   #pandas mean

mean_price_np = np.mean(data.price)



print("p mean : ", mean_point )

print ("np mean : ",mean_price_np )

data.describe()  #count, mean,std,min,max,%25,%50,%75 median values
data.corr()
# correlation map

f,ax = plt.subplots()

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt='.1f',ax=ax)
data.head(10)
data.columns
#line plot

data.points.plot(kind='line', color ='b', label = 'Points', linewidth=2, alpha = 0.5, grid = True, linestyle = ':')

data.price.plot(kind='line', color ='r', label = 'Price', linewidth=1, alpha = 0.5, grid = True, linestyle = '-.')

plt.legend(loc='upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

plt.show()
# scatter plot

# x = country, y = price

data.plot(kind = 'scatter', x='points',y='price', alpha=0.5,color='b', figsize = (3,3))

plt.xlabel('points')

plt.ylabel('price')

plt.title('Points price scatter plot')

plt.show()
# histogram

data.points.plot(kind='hist', bins=40,figsize = (3,3))

plt.show()
# clf() = cleans it up again you can start a fresh

data.points.plot(kind='hist',color='r', bins=40,figsize=(3,3))

plt.show()

plt.clf()
mydictionary = {'suzan' : 18, 'tugba' : 15}

print(mydictionary.keys())

print(mydictionary.values())

mydictionary['suzan'] = 26

print(mydictionary.values())

mydictionary['merve'] = 28

print(mydictionary.values())

print('kübra' in mydictionary)

mydictionary.clear()

print(mydictionary)

del mydictionary

print(mydictionary)
data = pd.read_csv('../input/winemag-data-130k-v2.csv')
series = data['price']

print(type(series))

data_frame = data[['price']]

print(type(data_frame))
x = data['price'] < 5

data[x]
data[np.logical_and(data['price']<10, data['points']>90)]  

data[(data['price']<10) & (data['points']>90)]    #same with previous code. "&" instead of "and"



data[np.logical_or(data['price']<10, data['points']>90)]

#while loop

i = 0

while i < 4 :

    print('i is:', i)

    i += 1

print(i, 'is equal to 4')
#for loop

list = [1,2,3,4,5,6]

for i in list : 

    print('i is:', i)

print('')



wantedname = 'name'

mylist = ['suzan', 'nazli','merve','tugba', 'kubra']

while wantedname != 'kubra' :

    for wantedname in mylist : 

        print('name is:', wantedname)

print(wantedname, 'bulundu')
for index, value in enumerate(list):

    print(index, ' : ', value)

print('')



dictionary = {'suzan' : 26, 'nazli' : 24, 'merve' : 28}

for key,value in dictionary.items():

    print(key, ":", value)

print('')



for index,value in data[['country']][2:4].iterrows():

    print(index, ":", value)