# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
myData = pd.read_csv("/kaggle/input/covid19-tweets/covid19_tweets.csv")
myData.head(10)

myData.corr()
f,ax = plt.subplots(figsize = (8,8))
sns.heatmap(myData.corr(), annot = True, linewidths = .5, fmt = '.1f', ax=ax)
plt.show()
#line plot
myData.user_followers.plot(kind="line",color = 'r',label='user_followers',linewidth = 1, alpha=0.5, grid = True, linestyle = '-.')
myData.user_friends.plot(color='y',label='user_friends',linewidth = 1, alpha=0.5, grid = True, linestyle = ':')
plt.legend(loc = 'upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line plot')
plt.show()
myData.plot(kind='scatter',x = 'user_followers',y = 'user_friends',alpha =.5, color='green')
plt.xlabel('user_followers')
plt.ylabel('user_friends')
plt.title('User Followers and Friends Scatter Plot')
myData.user_favourites.plot(kind='hist',bins=50,figsize=(8,8))
dictionary={'Turkey':'Istanbul'}
dictionary={'Turkey':'Istanbul','France':'Paris','Italy':'Roma'}
print(dictionary.keys())
print(dictionary.values())
dictionary={'Turkey':'Istanbul','France':'Paris','Italy':'Roma'}
print(dictionary)
dictionary['Turkey']="Ankara" #Update existing entry
print(dictionary)
dictionary['ABD'] = "New York" #add new entry
print(dictionary)
del dictionary['ABD'] #we can delete for not to fill memory
print(dictionary)
print('Turkey' in dictionary) #check exist or not
dictionary.clear() #remove all entries 
print(dictionary)
series = myData['user_friends']
print(type(series))
dataFrame = myData[['user_friends']]
print(type(dataFrame))
x = myData['user_friends']>2000
myData[x]
 #myData(myData['user_friends']>2000) &  myData(myData['user_favourites']>2000) #without using the np library
myData[np.logical_and(myData['user_friends']>2000, myData['user_favourites']>2000)]
a = 4
while a!=8:
    print(a)
    a += 1

x = [1,2,3]
for i in x: 
    print(i)
print('')

#Enumerate index and values of lists (e.g index:value, 0-1)
for index, value in enumerate(x):
    print(index,'-',value)
    
# for dictionaries. We can loop through a dictionary by using a for loop.
# When looping through a dictionary, the return value are the keys of the dictionary.

dictionary={'Turkey':'Istanbul','France':'Paris','Italy':'Roma'}
for key,value in dictionary.items():
    print(key,'-',value)
    
#We can achieve index and values (pandas)
#The iterrows() function is used to iterate over DataFrame rows as (index, Series) pairs.
#Iterates over the DataFrame columns, returning a tuple with the column name and the content as a Series.
for index,value in myData[['user_friends']][0:1].iterrows(): # [0:1] --> gets the first 
     print(key,'-',value)

def my_tuble():
    """ my tuble """
    tuble = (1,2,3)
    print(tuble)
    return tuble
x,y,z = my_tuble()
print(x,y,z)
a = 5
def func():
    a = 7
    return a
print(a)  #Global variable a = 5
print(func()) #Local variable a = 7
# We can see which ones built in scope
import builtins
dir(builtins)