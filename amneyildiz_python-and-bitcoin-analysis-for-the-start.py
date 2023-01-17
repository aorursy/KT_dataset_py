# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# We import the .csv file that we will analyze.

# We use the pandas library while importing the .csv file.

df = pd.read_csv('/kaggle/input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv')

# We display the first 5 data in the file with the head method.

df.head()
# We list the columns in our data with the information method.

# We access detailed suggestions on columns.

df.info()
#We access the values in the data.

df.corr()
#correlation map

f,ax = plt.subplots(figsize=(18,18))

#corr(),the correction method allows us to get all this into our data.

#annot allows displaying data values.annot=False,we can't see the values.

#linewidths,choos the lines as images between two values.

#fmt,it means how many values of the values after the comma display.

#for example= x=0,123 fmt=.1f x=0,1 or fmt=.3f x=0,123

sns.heatmap(df.corr(), annot=True, linewidths=.5,fmt='.1f',ax=ax) 

# we show() with the show

plt.show()
# head() method,look into the number displays in the list.

df.head(15)
# columns(),Shows the columns of the table in the dataset.

df.columns
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

df.Low.plot(kind = 'line', color = 'g',label = 'Low',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

df.High.plot(color = 'r',label = 'High',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot 

# x = Open, y = Close

df.plot(kind='scatter', x='Open', y='Close',alpha = 0.5,color = 'blue')

plt.xlabel('Open')              # label = name of label

plt.ylabel('Close')

plt.title('Attack Defense Scatter Plot')   

plt.show()
df.Open.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()
df.Close.plot(kind = 'hist',bins = 50,figsize = (12,12),color="red")

plt.show()
dictionary = {'Open' : 0.8,'Close' : 1.0, 'Day':'Monday'}

print(dictionary.keys())

print(dictionary.values())
dictionary['Day'] = "Sunday"    # update existing entry

print(dictionary)

dictionary['Year'] = 2020       # Add new entry

print(dictionary)

del dictionary['Year']              # remove entry with key 'spain'

print(dictionary)

print('Sunday' in dictionary)        # check include or not

dictionary.clear()                   # remove all entries in dict

print(dictionary)
# del dictionary         # delete entire dictionary     

print(dictionary)       # it gives error because dictionary is deleted
df = pd.read_csv('/kaggle/input/bitcoin-historical-data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv')
series = df['Open']        # data['Defense'] = series

print(type(series))

data_frame = df[['Open']]  # data[['Defense']] = data frame

print(type(data_frame))

# Comparison operator

print(0.1 > 2)

print(5!=2)

# Boolean operators

print(True and False)

print(True or False)
x = df['Open']>7000

df[x]



     

df[np.logical_and(df['Open']>9000, df['Close']>8000 )]
df[(df['High']>9000) & (df['Low']<9090)]

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



# For pandas we can achieve index and value

for index,value in df[['Open']][0:1].iterrows():

    print(index," : ",value)


