# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/countries of the world.csv')   # CSV: comma - separated values
data.info()
data.rename(columns={"Area (sq. mi.)": "Area", "Pop. Density (per sq. mi.)":"Pop_Density",
                        "Coastline (coast/area ratio)":"Coastline","Net migration":"Net_migration",
                        "Infant mortality (per 1000 births)":"Infant_mortality","GDP ($ per capita)":"GPD",
                        "Literacy (%)":"Literacy","Phones (per 1000)":"Phone_using","Arable (%)":"Arable",
                        "Crops (%)":"Crops","Other (%)":"Other"},inplace = True)
#We can be rename colums's name for to use easily.Because Python gives error at coding due to space.

data.columns
data.Literacy = data.Literacy.str.replace(",",".").astype(float)
data.Pop_Density = data.Pop_Density.str.replace(",",".").astype(float)
data.Coastline = data.Coastline.str.replace(",",".").astype(float)
data.Net_migration = data.Net_migration.str.replace(",",".").astype(float)
data.Infant_mortality = data.Infant_mortality.str.replace(",",".").astype(float)
data.Phone_using = data.Phone_using.str.replace(",",".").astype(float)
data.Arable = data.Arable.str.replace(",",".").astype(float)
data.Crops = data.Crops.str.replace(",",".").astype(float)
data.Birthrate = data.Birthrate.str.replace(",",".").astype(float)
data.Deathrate = data.Deathrate.str.replace(",",".").astype(float)
data.Agriculture = data.Agriculture.str.replace(",",".").astype(float)
data.Industry = data.Industry.str.replace(",",".").astype(float)
data.Service = data.Service.str.replace(",",".").astype(float)
data.Other = data.Other.str.replace(",",".").astype(float)
data.Climate = data.Climate.str.replace(",",".").astype(float)
data.dtypes
data.corr()   # correlation between features
#CORRELATION MAP
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()

#annot=True :It gives us correlation values inside the boxes
#linewidths= .5 :Thickness of line between boxes
#fmt= '.1f' :It gives how many will be written of correlation values after comma
data.head() #This coding gives us first 5 rows. If you want to see more lines, you should write a number inside the paranthesis.
data.columns  #This coding gives name of columns
#Line plot is better when x axis is time. 
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.Birthrate.plot(kind = 'line', color = 'g',label = 'Birthrate',linewidth=1,alpha = 0.5,grid = True,linestyle = '-')
data.Deathrate.plot(color = 'r',label = 'Deathrate',linewidth=1, alpha = 0.5,grid = True,linestyle = '-')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
#Scatter is better when there is correlation between two variables
# x = GPD, y = Industry
data.plot(kind='scatter', x='GPD', y='Industry',alpha = 0.5,color = 'red')
plt.xlabel('GPD')              # label = name of label
plt.ylabel('Industry')
plt.title('GPD-Industry Scatter Plot')            # title = title of plot
plt.show()
#Histogram is better when we need to see distribution of numerical data.
#bins = number of bar in figure
data.Infant_mortality.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
#clf = cleans it up again you can start a fresh
data.Phone_using.plot(kind = "hist", bins = 60, figsize = (10,10), grid = True)
plt.clf()
#we cant see plot due to clf
#create dictionary and look its keys and values
dictionary = {'Young_people' : '5500', 'Elderly_population' : '3300'}
print(dictionary.keys())
print(dictionary.values())
# Keys have to be immutable objects like string, boolean, float, integer or tubles
# List is not immutable
# Keys are unique
dictionary['Young_people'] = "7500"    # update existing entry
print(dictionary)
dictionary['Crime_rate '] = "150"      # Add new entry
print(dictionary)
del dictionary['Young_people']         # remove entry with key 'spain'
print(dictionary)
print('Young_people' in dictionary)          # check include or not
dictionary.clear()                     # remove all entries in dict
print(dictionary)
# In order to run all code you need to take comment this line
# del dictionary         # delete entire dictionary!    
print(dictionary)       # it gives error because dictionary is deleted
# To creat a series, use a square brackets! / To create a data frame, use two square brackets!
series = data['Area']        # data['Area'] = series
print(type(series))
data_frame = data[['Area']]  # data[['Defense']] = data frame
print(type(data_frame))
# Comparison operator
print(5 > 3)
print(3!=1)
# Boolean operators
print(True and False)
print(True or False)
x = data['Area']>10000000     # There are only 1 country who have higher area value than 10^7
data[x]
# There are only 6 countrys who have higher Area value than 1000000 and higher Pop_Density value than 50
data[np.logical_and(data['Area']>1000000, data['Pop_Density']>50 )]
# This is also same with previous code line. Therefore we can also use '&' for filtering.
data[(data['Area']>1000000) & (data['Pop_Density']>50 )]
# Stay in loop if condition( i is not equal 5) is true
i = 0
while i != 6 :
    print('i is: ',i)
    i +=2 
print(i,' is equal to 6')
# Stay in loop if condition( i is not equal 5) is true
lis = [1,2,3,4,5,6]
for i in lis:
    print('i is: ',i)
print('')

# Enumerate index and value of list
# index : value = 0:1, 1:2, 2:3, 3:4, 4:5, 5:6
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
for index, value in data[['Literacy']][0:3].iterrows():
    print (index,':',value)




