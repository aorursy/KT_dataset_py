# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/PlayerPersonalData.csv')
data.info()
data.corr()
f,ax = plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(),annot=True, linewidths=5, fmt='.1f',ax=ax)
plt.show()
data.head(10)
data.columns
data_f100 = data.head(100) #Took only the first 100 data 
#Line Plot
data_f100.Age.plot(kind='line',color='g',label='Age', linewidth=1,alpha=0.5,grid=True,linestyle=':')
data_f100.Potential.plot(color='r',label='Potential',linewidth=1, alpha = 0.5, grid= True, linestyle='-.')
plt.legend(loc='upper middle')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot')
plt.show()
# Scatter Plot
# x = potential, y = age
data_f100.plot(kind='scatter', x='Potential', y='Age', alpha = 0.5,color = 'green')
plt.xlabel('Potential')
plt.ylabel('Age')
plt.title('Potential and Age Relation in Football Players')
#Histogranm
#bins = number of bar in figure
data_f100.Potential.plot(kind='hist',bins=50,figsize=(12,12))
plt.show()
data.Potential.plot(kind = 'hist',bins = 50)
plt.clf()
#let's create a dictionary and look its keys and values
dictionary = {'Sasha':20,'Vlad':15}
print(dictionary.keys())
print(dictionary.values())
#Editing keys and values of the dictionary

dictionary['Sasha']=25   #update existing entry
print(dictionary)

dictionary['Ege']=6      #add new entry
print(dictionary)

del dictionary['Vlad']   #remove entry with key
print(dictionary)

print('Ege' in dictionary) #check include or not

dictionary.clear()       #remove all entries in dict
print(dictionary)
data = pd.read_csv('../input/PlayerPersonalData.csv')
series = data['Potential']
print(type(series))
data_frame = data[['Age']]
print(type(data_frame))
#1 - filtering pandas data frame 
x = data['Potential']>92

data[x]
#2 - filtering pandas with logical_and

data[np.logical_and(data['Potential']>92,data['Age']<24)]
#3 - filtering without a function does the same job with 2

data[(data['Potential']>92) & (data['Age']<24)]
i=0
while(i <= 5):
    print('i =',i)
    i +=1
print(i,'>= 5')
liste=[1,2,3,4,6]
for each in liste:
    print('each =',each)
print('')

for index,value in enumerate(liste):
    print(index,' : ', value)
print('')

dictionary = {'Sasha': 20,'Vlad': 15}
for key,value in dictionary.items():
    print(key,' : ',value)
print('')

for index,value in data[['Potential']][0:1].iterrows():
    print(index,' : ',value)

def tuble_ex():
    "return defined tuble"
    t = (1,2,3)
    return t
a = tuble_ex()
x,y,z = tuble_ex()
print('a =',a,' & Type of a=',type(a))
print('x =',x,' & Type of x=',type(x))
print('y =',y,' & Type of y=',type(y))
print('z =',z,' & Type of z=',type(z))
#nested function
def square():
    """ return square of value """
    def add():
        """ add two local variable """
        x = 2
        y = 3
        z = x + y
        return z
    return add()**2
print(square())   
name = "messi"
it = iter(name)
print(next(it))    # print next iteration
print(next(it))    # print next iteration
print(*it)         # print remaining iteration
# zip example
list1 = [1,3,5,7]
list2 = [2,4,6,8]
z = zip(list1,list2)
print(z)
z_list = list(z)
print(z_list)
print('')
# unzip z_list back to tuples
un_zip = zip(*z_list)
un_list1,un_list2 = list(un_zip) # unzip returns tuble
print(un_list1)
print(un_list2)
print(type(un_list2))
num=[2,4,6]
num2 = [i+1 for i in num]
print(num2)
threshold = sum(data.Age)/len(data.Age)
print(threshold)
data['Seniority'] = ['high'if each>threshold else 'low' for each in data.Age]
data.loc[:10,['Seniority','Age']]