# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns #visuliasion tool

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("../input/pokemon-challenge/pokemon.csv") # data seti path'i string olarak yazılmalı "" içinde

data.head()
data.info()
data.corr()
#correlation map (features arası ilişkiler)

f,ax = plt.subplots(figsize=(10,10)) #figure buyüklüğü

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt ='.1f',ax=ax) #

plt.show()
data.head(10)
data.columns
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.Speed.plot(kind = 'line', color = 'g',label = 'Speed',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.Defense.plot(color = 'r',label = 'Defense',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right') #legend=outs label into plot

plt.xlabel("x axis")          #label =name of label

plt.ylabel("y axis")

plt.title("Line plot")        #title = name of plot      

plt.show()

#Scatter Plot

#x=attack y=defense

data.plot(kind='scatter', x='Attack',y='Defense',alpha=0.5,color='red')

#plt.scatter(data.Attcak,data.Defense,color='red',alpha=0.5) //different method same result

plt.xlabel=('Attack')

plt.ylabel=('Defense')

plt.title=('Attack - Defense Scatter Plot')
#Histogram

# bins=number of ba figure

data.Speed.plot(kind ='hist',bins=50,figsize=(12,12))

plt.show()
# create a dictionary and look keys and values

dictionary={'spain' : 'madrid','usa':'vegas'}

print(dictionary.keys())

print(dictionary.values())
#keys have to be immutable object like string, boolean, integer or tubles

#list is not immutable

#keys are unique

dictionary['spain']='barcelona' #update existing entry

print(dictionary)

dictionary['france']='paris'  #add new entry

print(dictionary)

del dictionary['spain']     #remove enrtry with key spain

print(dictionary)

print('france' in dictionary)#check include or not

print('italy' in dictionary)

dictionary.clear()             #remove all entries

print(dictionary)

#in order run all code you need to take comment this line

#del dictionary  #delete entire dictionary

#print(dictionary) #it gives error because dictionary is deleted