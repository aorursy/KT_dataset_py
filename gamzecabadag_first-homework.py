# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/Iris.csv')

data.info()
data.corr()
#correlation map

#f,ax=plt.subplots()

#sns.heatmap(data.corr(),annot=True,linewidths=.5,fmt='.lf',ax=ax)

#Since this part doesn't work I added it as comment.
data.head(10)
data.columns

data.SepalWidthCm.plot(kind='line',color='g',label='SepalWidthCm',linewidth=1, alpha=0.5,grid=True)

data.SepalLengthCm.plot(color='r',label='SepalLengthCm',linewidth=1,grid=True,linestyle='-.')

plt.legend(loc='upper right')

plt.ylabel('y axis')

plt.title('Line Plot') #Title of plot

plt.show()
data.plot(kind='Scatter',x='SepalWidthCm',y='SepalLengthCm',alpha=0.5,grid=True,color='red')

plt.xlabel('SepalWidthCm')

plt.ylabel('SepalLengthCm')

plt.title('SepalWidthCm SepalLengthCm Scatter Plot')
#Histogram

#bins=number of bar in figure

data.SepalWidthCm.plot(kind='hist',bins=50)

plt.clf()
#create dictionary and loook its key and values

dictionary={'Spain':'Madrid','USA':'Vegas'}

print(dictionary.keys())

print(dictionary.values())

dictionary['Spain']="Barcelona"  #Update existing entry

print(dictionary)
dictionary['France']="Paris"

print(dictionary)
del dictionary['Spain']

print(dictionary)
print('France' in dictionary)
dictionary.clear()
data=pd.read_csv('../input/Iris.csv')

series=data['SepalLengthCm']

print(type(series))

data_frame=data[['SepalLengthCm']]

print(type(data_frame))
#Filtering Pandas Data Frame

x=data['SepalLengthCm']>5

data[x]
data[np.logical_and(data['SepalLengthCm']>5,data['PetalLengthCm']>5)]
data[(data['SepalLengthCm']>5) & (data['PetalLengthCm']>5)]
#While And For Loops

i=0

while i!=5:

    print('i is: ',i)

    i+=1

print(i,'i is equal to 5')    

    

    

    

    

    
lis=[1,2,3,4,5]

for i in lis:

    print('i is: ',i)

print('')

for index, value in enumerate(lis):

    print(index,":",value)

print('')    

dictionary={'spain':'madrid','france':'Paris'}

for key,value in dictionary.items():

    print(key, ":",value)

for index,value in data[['SepalLengthCm']][0:15].iterrows():

    print(index,":",value)
#My Homework is Ready