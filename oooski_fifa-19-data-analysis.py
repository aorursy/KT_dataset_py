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
data = pd.read_csv('../input/data.csv')
data.info()
data.corr()
# Correlation Map

f,ax = plt.subplots(figsize=(30,30))

sns.heatmap(data.corr(),annot=True , linewidths=.6, fmt = '.1f', ax = ax)

plt.show()
data.head(10)

data.columns # It gives us names of columns.
#Line Plot

data.Overall.plot(kind = 'line', color = 'r' , label = 'OVERALL' , linewidth = 2 , alpha = .8 , grid = True ,linestyle = ':',figsize=(15,15) )

data.Potential.plot(color = 'g' , label = 'POTENTIAL' , linewidth = .8 , alpha = .5 , linestyle = ':' ) 

plt.legend(loc = 'upper right')

plt.xlabel('Number of Players')

plt.ylabel('Overall Values')

plt.title('Line Plot Example')

plt.show()



#Scatter Plot

data.plot(kind = 'scatter' , x='Stamina' , y = 'SprintSpeed' , alpha = .5 , color = 'b',figsize=(15,15))

plt.xlabel('Stamina')

plt.ylabel('Sprint Speed')

plt.title('Stamina-Sprint Speed Scatter Plot')

plt.show()
#Histogram

data.SprintSpeed.plot(kind = 'hist' , bins = 100 , figsize = (20,20))

plt.xlabel('Sprint Speed')

plt.show()
# clf() = cleans it up again you can start a fresh

data.SprintSpeed.plot(kind = 'hist',bins = 50)

plt.clf()

dictionary = {'Spain': 'Madrid' , 'USA' : 'Vegas'}

print(dictionary.keys())

print(dictionary.values())
dictionary['Spain'] = 'Barcelona' # We updated the value of Spain key

print(dictionary)

dictionary['USA'] = 'Washington'# We updated the value of USA key

print(dictionary) 

dictionary['France'] = 'Paris' # We added a new key and value in dictionary

print(dictionary)

del dictionary['France'] # We deleted the key and value from dictionary

print(dictionary)

print('France' in dictionary) # If dictionary has the key named France, it will make turn True.

print('Spain' in dictionary) # This method checks only the keys !!

dictionary.clear()                   # remove all entries in dict

print(dictionary)





series = data['SprintSpeed'] # it gives us the parameters as vector that we wrote 

#print(series)

data_frame = data[['SprintSpeed']] # it gives us the parameters as data frame that we wrote 

print(data_frame)
# 1 - Filtering Pandas data frame

x = data['SprintSpeed']>95 # it gives us the values as True which  bigger than 95  

data[x] # It prints the True values as table

# 2 - Filtering pandas by using logical_and

data[np.logical_and(data['SprintSpeed']>95,data['Overall']>80)] # it gives us the values which SprintSpeed bigger than 95 and Overall bigger than 80



# This is also same with previous code line. Therefore we can also use '&' for filtering.

data[(data['SprintSpeed']>95) & (data['Overall']>80)]
# Stay in loop if condition( i is not equal 5) is true

i = 0

while i != 5 :

    print('i is: ',i)

    i +=1 

print(i,' is equal to 5')
# Stay in loop if condition( i is not equal 5) is true

lis = [1,2,3,4,5]

for i in lis:

    print('i is: ',i)

print('')



# Enumerate index and value of list

# index : value = 0:1, 1:2, 2:3, 3:4, 4:5

for index, value in enumerate(lis):

    print(index," : ",value)

print('')   



# For dictionaries

# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.

dictionary = {'spain':'madrid','france':'paris'}

for key,value in dictionary.items():

    print(key," : ",value)

print('')



 #For pandas we can achieve index and value

for index,value in data[['SprintSpeed']][0:2].iterrows():

    print(index," : ",value)




