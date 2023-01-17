# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns # visuallization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/fifa19/data.csv')
data.info() # We will learn something about this data set
data.corr() # We will make it better after Code Line

# We make with matplotlib
f,ax = plt.subplots(figsize = (18,18))

sns.heatmap(data.corr(), annot = True , linewidths = 1, fmt = '.1f',ax = ax)

plt.title('Correlation Map of Data Set')

plt.show()
data.head()

# Default value is 5 but you can change it like this
data.head(10)

# ".tail()" method also works in the same logic  
data.tail()
data.tail(10)
data.columns

# We can learn columns' names with this methods
# First of all we must fix the column names 

data.columns = [each.split()[0]+'_'+each.split()[1] if(len(each.split()) > 1) else each for each in data.columns]

#This method combines spaces between words

data.columns = [each.lower() for each in data.columns]

#This method writes the titles in lower case

#We will learn for loop later in this kernel
data.columns

# This looks better
# Line Plot 

data.overall.plot(kind = 'line',color = 'red',label = "Overall",linewidth = 0.75,alpha = 1,grid = True)

data.potential.plot(color = 'green',label = 'Potential',linewidth = 1,grid = True,alpha = 0.15,linestyle = '-.')

plt.legend(loc = 'upper right')

plt.xlabel('X Axis')

plt.ylabel('Y Axis')

plt.title('Example of Line Plot')

plt.show()
#Scatter Plot

data.plot(kind = 'scatter', x = 'acceleration',y = 'sprintspeed',alpha = 0.25,color = 'red')

plt.xlabel('Acceleration')

plt.ylabel('Sprintspeed')

plt.title('Acceleration and Sprint Speed Scatter Plot')

plt.show()
#Histogram Plot

# bins = number of bar in figure

data.jersey_number.plot(kind = 'hist',bins = 99,figsize = (10,10))

plt.xlabel('Players Jersey Number')

plt.ylabel('Frequency')

plt.show()

# bins is 99 because 99 kind jersey number we have
#Let's create a dictionary!

dictionary = {'fruit' : 'apple','vegetable' : 'carrot'}

# fruit and vegetable are "KEY"

# apple and carrot are "VALUE"

print(dictionary.keys())

print(dictionary.values())
dictionary['fruit'] = 'banana' # update

print(dictionary)

dictionary['fast-food'] = 'hamburger' # new entry

print(dictionary)

del dictionary['fruit'] # remove entry with key 'fruit'

print('fast-food' in dictionary) # cheack include or not

dictionary.clear() #remove all entries in dictionary

print(dictionary)
# Comparison operator



print(156 > -1)

print(1000000000 != 0)



# Boolean operators

print(True and False)

print(True or False)
x = data.gkreflexes > 88

# We have a 7 player who have higher goalkeeper reflexes value than 90

data[x]
# Let's use one Boolean operator

data[np.logical_and(data.gkreflexes > 88,data.overall >= 90)]

#You can only use 2 comparison options
# You can use Code line too

data[(data.gkreflexes > 88) & (data.overall >= 90)]
n = 0

while n != 5:

    print('n is : ',n)

    n += 1

print(n,' is equal to 5')    
array1 = np.arange(1,6,1)# You don't need this method but this is same logic with lis = [1,2,3,4,5] 

lis = list(array1)

for n in lis:

    print('n is : ',n)

print(' ')    
# Enumerate index and value of list

# index : value = 0:1, 1:2, 2:3, 3:4, 4:5

for index, value in enumerate(lis):

    print(index," : ",value)

print('')   
# For dictionaries

# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.

dictionary = {'fruit' : 'apple','vegetable' : 'carrot'}

for key,value in dictionary.items():

    print(key," : ",value)

print('')
# For pandas we can achieve index and value

for index,value in data[['overall']][0:1].iterrows():

    print(index, ' : ' ,value)
data = pd.read_csv("/kaggle/input/fifa19/data.csv")

data.head() # We learned this code line the starting of this kernel
data.tail() # This too
# shape gives number of rows and columns in a tuble

data.shape
# info gives data type like dataframe, number of sample or row, number of feature or column, feature types and memory usage

data.info()
data.columns = [each.split()[0]+"_"+each.split()[1] if(len(each.split()) > 1) else each for each in data.columns]

data.columns = [each.lower() for each in data.columns]

data.columns
data_head = data.head(125)

a = data_head.nationality.value_counts(dropna = False)

a
data.describe()
data.boxplot(column = "stamina",by = "work_rate",figsize = (12,12))

plt.show()
data_new = data.head(10)

data_new
data.columns
# Firstly I create new data from players data to explain melt more easily.

# lets melt

# id_vars = what we do not wish to melt

# value_vars = what we want to melt

melted = pd.melt(frame = data_new,id_vars = "name",value_vars = ["age","work_rate","value","wage"])

melted
# Index is name

# I want to make that columns are variable

# Finally values in columns are value



pivoted = melted.pivot(index = "name",columns = "variable",values = "value")

pivoted
# We can concatenate two dataframe

# Firstly lets create 2 data frame



data_head = data.head()

data_tail = data.tail()



conc_data_row = pd.concat([data_head,data_tail],axis = 0,ignore_index = True)

conc_data_row
data1 = data.overall.head()

data2 = data.potential.head()



data_1_and_2_conc = pd.concat([data1,data2],axis = 1)

# axis = 0 : adds dataframes in row

data_1_and_2_conc
data.dtypes
# lets convert object(str) to categorical and int to float

data.nationality = data.nationality.astype("category")

data.overall = data.overall.astype("float")
# As you can see Type 1 is converted from object to categorical

# And Speed ,s converted from int to float

data.dtypes
# Lets look at does player data have nan value

# As you can see there are 18207 entries. However release_clause has 16643 non-null object so it has 1564 null object.

data.info()
data.release_clause.value_counts(dropna = False)
data.release_clause.dropna(inplace = True) # inplace = True means we do not assign it to new variable. Changes automatically assigned to data

# So does it work ?
#  Lets check with assert statement

# Assert statement:

assert 1==1 # return nothing because it is true
# In order to run all code, we need to make this line comment

# assert 1==2 # return error because it is false
assert data.release_clause.notnull().all() # returns nothing because we drop nan values
data.release_clause.fillna('empty',inplace = True)
assert data.release_clause.notnull().all()# returns nothing because we do not have nan values