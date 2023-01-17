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
data = pd.read_csv('../input/AppleStore.csv')
#some information about track_name variable

data['track_name'].describe()
data.info()
data.corr()
#correlation map

f = plt.subplots(figsize=(15,12))

sns.heatmap(data.corr(), annot= True, linewidths=0.3, fmt='0.1f')

plt.show()
#if you don't enter the parameter. The first 5 elements are shown

data.head(7)
data.columns
#Line Plot

data.price.plot(kind = 'line', color = 'g',label = 'Price',linewidth = 1, alpha = 0.8, grid = True, linestyle = ':', figsize = (12,8))

data.user_rating.plot(color = 'b', label = 'User Rating', linewidth = 1, alpha = 1, grid = True, linestyle = '-')

plt.legend(loc= 'upper right')

plt.axis([3000, 3500, 0, 50])      # x axis = 3000~3500, y axis = 0~50

plt.xlabel('Applications in App Store')

plt.ylabel('Value')

plt.title('Line Plot')             # title = title of plot

plt.show()
#Scatter Plot 

#x = price, y = user_rating

data.plot(kind = 'scatter', x = 'price', y='user_rating', alpha = 0.5, color = 'green', figsize = (12,8))

plt.xlabel('Price')

plt.ylabel('User Rating')

plt.title('Price - User Rating Scatter Plot')

plt.show()
#Histogram

#bins = number of bar in plot

data.user_rating.plot(kind = 'hist', color = 'red', bins = 50, figsize = (12,8))

plt.show()
#clf() = cleans it up

data.user_rating.plot(kind = 'hist', bins = 50)       #we'll not see this plot

plt.clf()
#Natural satellite numbers of planets

dictionary = {'saturn' : 62, 'jupiter' : 67, 'earth' : 1}

print(dictionary.keys())

print(dictionary.values())
#Keys must be unique

dictionary['jupiter'] = 79           #update existing entry

print(dictionary)

dictionary['uranus'] = 27            #add new entry

print(dictionary)

del dictionary['earth']              #remove entry with key 'earth'

print(dictionary)

print('saturn' in dictionary)        #check include or not

dictionary.clear()                   #remove all entries in dictionary

print(dictionary)
#We can delete the dictionary if we want

#del dictionary          #delete dictionary     

print(dictionary)       #it gives error because dictionary is deleted
data = pd.read_csv('../input/AppleStore.csv')
series = data['size_bytes']        # data['Defense'] = series

print(type(series))

data_frame = data[['size_bytes']]  # data[['Defense']] = data frame

print(type(data_frame))
#Comparison operator

print(11 > 7)

print(5 != 9)

print(33 == 33)

#Boolean operators

print(True and False)

print(True or False)
#Filtering

x = data['size_bytes'] > 3800000000     # There are 7 applications who have higher size_bytes value than 3800000000

data[x]
#Filtering with Logic AND Gate

# There are 4 applications who have higher size_bytes value than 3800000000 and higher user_rating value than 4

data[np.logical_and(data['size_bytes'] > 3800000000, data['user_rating']> 4 )]
#Same as previous code, we just used the '&' sign

data[(data['size_bytes'] > 3800000000) & (data['user_rating'] > 4)]
#while loop

state, i = True, 0      #state = True, i = 0

while (state):          #while loop continues as long as the 'state' value is True

    print('i = ', i)

    i += 1

    state = (i != 5)

print('latest data = ', i)
#for loop

for i in range(2, 7):

    print('i = ', i)

print('')
#looping over a list

food_list = ['pasta', 'sushi', 'kebab', 'pizza', 'taco']

for i in food_list:

    print('i = ', i)

print('')



#which index corresponds to which value

for index, value in enumerate(food_list):

    print(index, ' = ', value)
dictionary = {'saturn': 62,'jupiter': 79}

for key,value in dictionary.items():

    print(key, ' = ', value)

print('')



for index,value in data[['cont_rating']][0:1].iterrows():     # first data

    print(index," : ", value)