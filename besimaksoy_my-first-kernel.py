# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import datetime

import os



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



print("Data listing...")

print(os.listdir('../input'))



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



        print('Using bitstampUSD_1-min_data...')

# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv")

df.head()
print( df.isnull().sum())
df['Volume_(BTC)'].fillna(value=0, inplace=True)

df['Volume_(Currency)'].fillna(value=0, inplace=True)

df['Weighted_Price'].fillna(value=0, inplace=True)





df['Open'].fillna(method='ffill', inplace=True)

df['High'].fillna(method='ffill', inplace=True)

df['Low'].fillna(method='ffill', inplace=True)

df['Close'].fillna(method='ffill', inplace=True)



print(df.tail())
df.info()

df.corr()
#correlation map

f,ax = plt.subplots(figsize=(11, 11))

sns.heatmap(df.corr(), annot=True, linewidths=.9, fmt= '.1f',ax=ax)

df.head(15)
df.columns
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

df.Timestamp.plot(kind = 'line', color = 'g',label = 'Timestamp',linewidth=7,alpha = 1.0,grid = True,linestyle = ':')

df.Weighted_Price.plot(color = 'r',label = 'Weighted_Price',linewidth=6, alpha = 1.0,grid = True,linestyle = '-.')

plt.legend(loc='center right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            

# Scatter Plot

# x = Timestamp, y = Weighted_Price

df.plot(kind='scatter', x='Timestamp', y='Weighted_Price',alpha = 0.5,color = 'purple')

plt.xlabel('Timestamp')              # label = name of label

plt.ylabel('Weighted_Price')

plt.title('Timestamp Weighted_Price Scatter Plot')            # title = title of plot
# PLOTS

fig = plt.figure(figsize=[15, 7])

plt.suptitle('Bitcoin exchanges, mean USD', fontsize=22)



plt.subplot(221)

plt.plot(df.Weighted_Price, '-', label='By Days')

plt.legend()



dictionary = {'turkey' : 'sinop','russia' : 'moscow'}

print(dictionary.keys())

print(dictionary.values())
dictionary['turkey'] = "sinop"    # update existing entry

print(dictionary)

dictionary['russia'] = "moscow"       # Add new entry

print(dictionary)



del dictionary['russia']              # remove entry with key 'spain'

print(dictionary)



print('usa' in dictionary)   

# check include or not

dictionary.clear()                   # remove all entries in dict

print(dictionary)
print(dictionary)   
data = pd.read_csv('/kaggle/input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv')
series = data['Volume_(BTC)']        # data['Volume_(BTC)'] = series

print(type(series))

data_frame = data[['Volume_(BTC)']]  # data[['Volume_(BTC)']] = data frame

print(type(data_frame))
# Comparison operator

print(5 > 2)

print(4!=2)



# Boolean operators

print(True and False)

print(True or False)
x = data['Volume_(BTC)']>200     

data[x]
data[np.logical_and(data['Volume_(BTC)']>200, data['Weighted_Price']>100 )]
data[(data['Open']>200) & (data['Close']>100)]
i = 0

while i != 15 :

    print('i is: ',i)

    i +=1 

print(i,' is equal to 15')
# Stay in loop if condition( i is not equal 15) is true

lis = [1,2,3,4,5,6,8,9,10,11,12,13,14,15]

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

dictionary = {'turkey':'sinop','russia':'moscow'}

for key,value in dictionary.items():

    print(key," : ",value)

print('')



# For pandas we can achieve index and value

for index,value in data[['Open']][0:1].iterrows():

    print(index," : ",value)