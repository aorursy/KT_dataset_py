# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt

import seaborn as sns 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/iris/Iris.csv')

# Importing data


data.info()

#general info about data
f,ax=plt.subplots(figsize=(18,18))

sns.heatmap(data.corr(),annot=True,linewidths=.5,fmt='.1f',ax=ax)

# Visualising of data 
data.corr()

# As we see from correlation table mostly there is no relation between the variables,or just makes inverse correlation.

# There is no Relation between PetalWidthCm and PetalLengthCm ,

# because correlation coefficient is 0. We can consider strong and weak

# relationships accordingly for correlation coefficient which is more and less than 1.
data.head(10)

# data of species of first 10  variables
data.SepalLengthCm.plot(kind = 'line', color = 'g',label = 'SepalLengthCm',markersize=5,marker='o',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.PetalLengthCm.plot(color = 'r',label = 'PetalLengthCm',linewidth=1, markersize=5,marker='*',alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     

plt.xlabel('x axis')              

plt.ylabel('y axis')

plt.title('Line Plot')            

plt.show() 

# Red line represents PetalLengthCm 

# Green line represents SepalLengthCm 

# x axis represents Iris

# y axis represents lengths of Petal and Sepal
data.plot(kind='scatter', x='Species', y='PetalLengthCm',alpha = 0.5,color = 'red')

plt.xlabel('Species')              # label = name of label

plt.ylabel('PetalLengthCm')

plt.title('Species PetalLengthCm Plot')            

# as we have no relation between PetalLengthCM and Species we see each 

#Setosa,Virginica and Versicolor  are increasing due their length constantly and independently
data.PetalLengthCm.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()

# x axis represents PetalLengthCm

# y axis represents frequences 

# We can identify from graph that we only have 2 species which have 6 cm length approximately
dictionary = {'AZERBAIJAN' : 'BAKU','GERMANY' : 'BERLIN'}

print(dictionary.keys())

print(dictionary.values())

# creating a dictionary
dictionary['AZERBAIJAN']='QUBA'

print (dictionary)

del dictionary['AZERBAIJAN']

print (dictionary)

#Replace Baku with Quba
print ('ITALY' in dictionary)

dictionary.clear()

print(dictionary)

# check if there exists ITALY in our dictionary
series = data['SepalWidthCm']        

print(type(series))

data_frame = data[['SepalWidthCm']]  

print(type(data_frame))

x=data['SepalLengthCm']>3

data[x]

# show all data where sepallength is more than 3
data[np.logical_and(data['SepalLengthCm']>2, data['PetalLengthCm']>3 )]

# show data where sepallength is more than 2 and petallength is more 3
data[(data['SepalLengthCm']>2) & (data['PetalLengthCm']>3)]
i = 0

while i != 4 :

    print('i is: ',i)

    i +=1

print(i,' is equal to 4')

# Usage of While Loop
lis = [1,2,3,4,5]

for i in lis:

    print('i is: ',i)

print('')

#give values of list
for index, value in enumerate(lis):

    print(index," : ",value)

print('')

# enumarate helps us to write values with their index
dictionary = {'azerbaijan':'baku','france':'paris'}

for key,value in dictionary.items():

    print(key," : ",value)

print('')
for index,value in data[['PetalLengthCm']][0:1].iterrows():

    print(index," : ",value)

    #gives both data and value of where petallengthcm is from 0-1 interval