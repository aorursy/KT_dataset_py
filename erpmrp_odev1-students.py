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
#import necessary addtitional libraries:

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool
#import the data into data_Students

data_Students = pd.read_csv("../input/students-performance-in-exams/StudentsPerformance.csv")
#check the headers:

data_Students.info()
#check the firs 10 records.

data_Students.head(10)
#try to arrange the column names

#first time worked, then did not work:

data_Students.rename(columns = {'math score':'mathscore'})

data_Students.head(10)
#removes spaces form the columns:

data_Students = data_Students.rename(columns={c: c.replace(' ', '') for c in data_Students.columns})
#check:

data_Students.head(10)
# correlations



data_Students.corr()
# correlation map (heat map)

f, ax = plt.subplots(figsize=(18,18)) #çerçeveyi çizer

sns.heatmap(data_Students.corr(), annot=True, linewidth=.5, fmt='.2f', ax=ax)

plt.show()
#line plot



data_Students.readingscore.plot(kind='line', color='g', label='Reading', linewidth=1,alpha=0.7, grid=True, linestyle=':')

#be careful for the linestyle, doesn't support charactes like '..'

#x axis shows numbers, like indexes. which number of the student in the list



#numerik değerler seçilmeli,#numerik değerler seçilmeli,

data_Students.mathscore.plot(color='r', label='Math', linewidth=1, alpha=0.7, grid =True, linestyle='-.')



plt.legend(loc='upper right')



plt.xlabel('x axis Reading scores')

plt.ylabel('y axis')

plt.title('Line Plot scores')

plt.show()

# check basic statistics:

data_Students.describe()
#scatter plot1-1

plt.scatter(data_Students.mathscore, data_Students.readingscore,color='red', alpha=0.5) 

plt.xlabel('Reading')

plt.ylabel('Math')

plt.title('Reading Math Scatter')

plt.show()
#histogram-1

data_Students.plot(kind='hist', bins=30,figsize=(12,15))

plt.show()
#histogram-2

data_Students.mathscore.plot(kind='hist', bins=30,figsize=(12,15))

plt.show()
#histogram-3

data_Students.mathscore.plot(kind='hist', bins=50)

plt.show()
dic= {'usa':'boston', 'spain': 'barcelona'}

print(dic.keys())
print(dic.values())
#update records:

dic['spain'] = 'valencia'

print(dic)
#add new entry:

dic['france'] = "nice"

print(dic)
#remove entry with key spain:

del dic['spain']

print(dic)
#check includes or not

print('france' in dic)
#clear dictionary

dic.clear()

print(dic)



#to delete permanently from memory: del dic
data_Students.info()
data_Students.head(10)
# basic statistics



data_Students.describe()
# create a series. gender from data_Students

seriSt = data_Students['gender']

print(type(seriSt))
print(seriSt)
#move to a dataframe

dframeSt = data_Students[['gender']]

print(dframeSt)
#comparasion

2==3
print(5!=7)
# Boolean

print(5!=5 and 3==3)


print(6==6 or 8==9)
data_Students.head(10)
# 1-filtering

x = data_Students['readingscore']>97

data_Students[x]
# 1-filtering with logical_and 

data_Students[np.logical_and (data_Students['mathscore']>96, data_Students['readingscore']>95)]
# another way to write:

data_Students[(data_Students['mathscore']>96) & (data_Students['readingscore']>95)]
# i=10 a kadar ekrana yazdır:



i = 0

while i<10 :

    print ('i: ', i)

    i +=1

    

print('loop tan sonraki son değer:', i)
# create a list, 

mylist = [10,20, 300, 4, 500]



# print the items

for i in mylist:

    print('i değeri: ', i)

    

print('stop')
# enumerate, reach with index numbers



for indx, i in enumerate(mylist):

    print('index no:', indx, "value:", i)

    

print("stop")
# reach dictionary values, get key and values

#use for loop

dic2 = {'spain':'madrid', 'france':'paris'}



for dkey, dvalue in dic2.items():

    print("dkeyx:", dkey, "dvaluex:", dvalue)

    

print("stop")
data_Students.head(2)
# reach index and values, pandas

# records with index numbers 0,1 



for ind, val in data_Students[['mathscore']] [0:2].iterrows():

    print('indeks:', ind,'değer:',val )