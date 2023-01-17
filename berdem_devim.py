import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

#We are bringing the dataframes to the screen.
#heart.csv is coming to the screen

data = pd.read_csv('../input/heart.csv')

# Uploading survey.csv data
#examining dataframe structure

data.info()

#as a result float64(1), int64(13)
data.corr()

#I can't wait to see the correlation similarities between them.
#correlation map

f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(data.corr(), annot=True, linewidths=.4, fmt= '.1f',ax=ax)

#We use the seoborn library. (sns) first parameter data.corr ,

#https://seaborn.pydata.org/generated/seaborn.heatmap.html



plt.show()
#to get the first 10 lines

data.head(10)
data.columns

#to retrieve column names
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.cp.plot(kind = 'line', color = 'g',label = 'chest pain',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.thalach.plot(color = 'r',label = 'maximum heart rate',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

data.age.plot(color = 'b',label = 'age',linewidth=0.5, alpha = 0.8,grid = True,linestyle = 'dashed')





plt.legend(loc='upper right', bbox_to_anchor=(1, 0.3))     # legend = puts label into plot



#https://matplotlib.org/api/legend_api.html





plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot 

# x = attack, y = defense

data.plot(kind='scatter', x='thalach', y='slope',alpha = 0.8,color = 'brown')

plt.xlabel('thalach')              # label = name of label

plt.ylabel('slope')

plt.title('The slope of the peak exercise ST segment maximum heart rate achieved Scatter Plot')  

plt.show()# title = title of plot

#here's the correlation
# Scatter Plot 

# x = attack, y = defense

#data.plot(kind='scatter', x='thalach', y='age',alpha = 0.8)

plt.scatter(data.thalach,data.age,alpha=0.4,color = 'green')

plt.xlabel('thalach')              # label = name of label

plt.ylabel('age')

plt.title('Age and maximum heart rate achieved Scatter Plot')  

plt.show()# title = title of plot

#here's not the correlation
# Histogram

# bins = number of bar in figure

data.chol.plot(kind = 'hist',bins = 40,figsize = (6,6),alpha=0.5,layout ='tuple',sharex=False)

plt.show()

#http://pandas.pydata.org/pandas-docs/version/0.19.0/generated/pandas.DataFrame.hist.html
#create dictionary and look its keys and values

dictionary = {'Name' : 'Bülent','Age' : 53,'City':'Adana','Test':1}

print(dictionary.keys())

print(dictionary.values())
# Keys have to be immutable objects like string, boolean, float, integer or tubles

# List is not immutable

# Keys are unique

dictionary['City'] = "İstanbul"    # update existing entry

print(dictionary)

dictionary['Country'] = "Turkey"       # Add new entry

print(dictionary)

del dictionary['Test']              # remove entry with key 'spain'

print(dictionary)

print('Country' in dictionary)        # check include or not

dictionary.clear()                   # remove all entries in dict

print(dictionary)
# In order to run all code you need to take comment this line

# del dictionary         # delete entire dictionary     

print(dictionary)       # it gives error because dictionary is deleted
data = pd.read_csv('../input/heart.csv')



series = data['chol']        # data['Defense'] = series

print(type(series))

data_frame = data[['chol']]  # data[['Defense']] = data frame

print(type(data_frame))
# 1 - Filtering Pandas data frame

x = data['chol']>200     # There are only 252 person who have higher cholestrol value than 200

data[x]
# 2 - Filtering pandas with logical_and

# There are only 5 person who have higher cholestrol value than 270 and pain value equal 3

data[np.logical_and(data['chol']>270, data['cp']==3 )]
# This is also same with previous code line. Therefore we can also use '&' for filtering.

data[(data['chol']>280) & (data['cp']==3)]
i = 0

while (i <5):

   print ('The count is:', i)

   i = i + 1



print ("Bye bye!")
lis = [10,20,30,40,50]

for i in lis:

    print('i is: ',i)

print('')



# Enumerate index and value of list

# index : value = 0:10, 1:20, 2:30, 30:4, 40:50

for index, value in enumerate(lis):

    print(index," : ",value)

print('')   



# For dictionaries

# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.

dictionary = {'Name' : 'Bülent','Age' : 53,'City':'Adana','Test':1}

for key,value in dictionary.items():

    print(key," : ",value)

print('')



# For pandas we can achieve index and value

for index,value in data[['chol']][0:1].iterrows():

    print(index," : ",value)