# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data plotting - I added this line 
import seaborn as sns # for data visualizing - I added this line
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

from subprocess import check_output # this line is taken exactly as it is from the tutor's kernel
print(check_output(["ls", "../input"]).decode("utf8")) # this line is taken exactly as it is from the tutor's kernel

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/creditcard.csv') # here the dataset is read by using "pd.read_csv"
data.info()
data.columns
data.corr()
# correlation map of the data
f,ax = plt.subplots(figsize=(15, 15)) # adjusting the figure size here
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax) 
plt.show()
data.head(10) # checking the first ten rows of the data
# Line plot 
data.V1.plot(kind = 'line', color = 'r', label = 'V1', figsize = (15,15), linewidth = 1, alpha = 0.7, grid = True, linestyle = ':')
data.V2.plot(color = 'g', label = 'V2', linewidth = 1, alpha = 0.7, grid = True, linestyle = '-.')
plt.legend(loc = 'lower right') # locates the legend
plt.xlabel('transactions')     # x-axis stands for the transactions
plt.ylabel('V-Parameters')     # y-axis stands for the V1 and V2 parameters
plt.title('Line Plot of V1 and V2 versus transactions')
plt.show()
# Scatter Plot
# x = Amount , y = V2
data.plot(kind = 'scatter', x = 'V2', y = 'Amount', figsize = (18,18), grid = True, alpha = '0.6', color = 'b')
plt.xlabel('V2')
plt.ylabel('Amount')
plt.title('V2 - Amount Scatter Plot')
plt.show()


# Histogram
# here we will take a look at the frequency of 'Time' vaues.
data.Time.plot(kind = 'hist', bins = 1000, figsize = (18,18), grid = True)
plt.show()
# Here I would like to try the clf() command which cleans the plot up.
data.Time.plot(kind = 'hist', bins = 100, figsize = (12,12))
plt.clf()
# We will not be able to see the histogram due to the clf() command.
water_molecule = {'hydrogen': '2', 'oxygen': '1'}
print(water_molecule.keys())
print(water_molecule.values())
print(water_molecule)
data.head() # I will choose the keys and values from here

#dictionary_ccfd = {'data.columns[0]': 'data.columns[0][2]', 'data.columns[29]': 'data.columns[29][2]', 'data.columns[30]': 'data.columns[30][2]'}
dictionary_ccfd = {data.columns[0]: data.Time[2], data.columns[29]: data.Amount[2], data.columns[30]: data.Class[2]}
print(dictionary_ccfd)
print(dictionary_ccfd.keys())
print(dictionary_ccfd.values())

# with this example I realized that when calling keys and values from an existing data set, we do not need 
# to use ' ' such as 'data.columns[0]', using data.columns[0] is okay.
# As we created our dictionary using the data set,  now we can try some properties of dictionaries.

dictionary_ccfd[data.columns[0]] = data.Time[4]   # update existing value
print(dictionary_ccfd)
dictionary_ccfd[data.columns[1]] = data.V1[2]     # add new entry
print(dictionary_ccfd)
del dictionary_ccfd[data.columns[0]]              # remove entry with key 'Time'
print(dictionary_ccfd)
print('Amount' in dictionary_ccfd)                # check if Amount is included in the dictionary_ccfd
dictionary_ccfd.clear()                           # clear all the entries of the dictionary_ccfd
print(dictionary_ccfd)

# in order to delete the dictionary_ccfd totally we need to use the del function
#del dictionary_ccfd                             # delete entire dictionary
print(dictionary_ccfd)   # it gives the name error : NameError: name 'dictionary_ccfd' is not defined

data = pd.read_csv('../input/creditcard.csv')
series = data['Time']                # data['Time'] = series
print(type(series))                  # check the type of the series
data_frame = data[['Time']]          # data[['Time']] = data_frame
print(type(data_frame))
# comparison operators
print(data.Time[3] > data.Time[30])              # compare the values in a specific column
print(data.Amount[5] == data.Amount[50])

# I would like to check if I can add comparison action in a dictionary

dictionary_comp = {'data.Time[3]': data.Time[3], 'data.Time[30]': data.Time[30], 'comparison result (value1>value2)': data.Time[3] > data.Time[30]}
print(dictionary_comp)

# boolean operators

print(data.Time[3] > data.Time[30] or data.Amount[5] != data.Amount[50])
print(data.Time[0] == data.Time[1] and data.Time[2] == data.Time[3])
# Filtering pandas

x = data['Class'] == 1                          # trying to find the number of fraudulent transactions
data[x]
# now let's filter the data with more than one condition as follows:

#y = data['Time'] > 165000 
#z = data['Class'] == 1
#data[y & z]                               
# this is the method what we used in the previous example

data[np.logical_and(data['Time'] > 160000, data['Class'] == 1)]  

# this is the method using numpy's np.logical_and operator
data[np.logical_and(data['Time'] > 160000, data['Class'] == 1)].describe()
#data[np.logical_and(data['Time'] > 160000, data['Class'] == 1)].info
# we can see info (count, mean etc.) of the result by using .describe() or we can also use .info to see some basic information  
# the below line also gives the same result with np.logical_and operator, we see that we can also use '&'
data[(data['Time'] > 160000) & (data['Class'] == 1)]
# Stay in loop if condition is true
# condition : i is equal to 1
# I would like to learn the occurance of the first fraudulent transaction. How can I do that?
# Let me try as follows:

i = 0
while data.Class[i] != 1 :
    #print('Class is: 0')
    i += 1
print(i,'is the ranking of the first occurance of the fraudulent transaction.')
# give an alert if the condition is true
# condition : i is equal to 1
# I would like to create a similar loop that we created in the previous example
# I will investigate a small portion of the data

sigma = data.Class[530:550]
for i in sigma:
    if i == 1:
        print('A fraudulent transaction is detected.')
print('')
# Let's try to enumerate index ad value of a list
# index : value = 0:0, 1:0 etc.

for index, value in enumerate(sigma):
    print(index,':',value)
print('')
# For dictionaries we can also use for loops in order to find a keys and values of the dictionary
# Let's define a new dictionary as follows:

dictionary_new = {data.columns[3]:data.V3[541], data.columns[4]:data.V4[541], data.columns[30]:data.Class[541]}
print(dictionary_new)
# Now I can create a for loop

for key,value in dictionary_new.items():
    print(key,':',value)
print('')
# For pandas we can also find the index and value by using for loop

for index,value in data[['Amount']][0:1].iterrows():
    print(index,':',value)
print('')
# we will try to practice tuble and docstring as follows:

def tuble_time():
    """define a function of 'Time' column of our data set"""    # this is the docstring of the function
    t = (data.Time[0],data.Time[1],data.Time[2])
    return t
a,b,c = tuble_time()
print(a,b,c)
# difference between local and global scopes

a = 5
def f():
    a = 10
    return a
print(a)                     # global scope
print(f())                  # local scope
# Let's see what happens when there is no local scope

a = 10
def f():
    b = a**2               # there is no local scope a
    return b
print(f())                 # it uses global scope a

# The script first searches local scope, then global scopes. 
# If nothing found in these then script searches for built in scopes. 
# Let's see how we can see the built-in scopes:

import builtins
dir(builtins)
# nested function

def divide():
    """this function returns the one tenth of the value"""
    def subtract():
        """this function returns the subtraction of 10 from the value"""
        c = 200
        d = c-10
        return d
    return subtract()/10
print(divide())
# Default arguments

def g(m,n=5,o=25):
    p = m+n+o
    return p
print(g(1))

# If we would like to change the default arguments, we can simply enter the variables to the function

print(g(1,20,40))
# flexible arguments

def h(*args):
    for i in args:
        print(i)
h(1)
print("")
h(1,2,3,4)

# flexible arguments **kwargs which is dictionary

def fg(**kwargs):
    """print keys and values of the dictionary"""
    for key, value in kwargs.items():
        print(key,":",value)
fg(atom1 = 'hydrogen', atom2 = 'oxygen', molecule = 'water', molecule_weight = 18)


# lamnda function

cube = lambda x: x**3
print(cube(5))

total = lambda y,z,t: y+z+t
print(total(4,6,8))
numbers = [3,6,9]
t = map(lambda x: x**3,numbers)
print(list(t))
# iterators example

name = "kaggle"
it = iter(name)
print(next(it))                     # print the iteration
print(next(it))                     # print the next item in the iteration
print(*it)                          # print the rest of the iteration


# zip function is used for combining the lists

liste1 = [5,10,15]
liste2 = [20,25,30]

zed = zip(liste1,liste2)
print(zed)
zed_liste = list(zed)
print(zed_liste)
# zip(*...) seperates the zipped lists

unzip1 = zip(*zed_liste)
un_liste1,un_liste2 = list(unzip1)
print(un_liste1)
print(un_liste2)
print(type(un_liste2))


# example of list comprehension
num1 = [1,2,3]
num2 = [i+1 for i in num1]
print(num2)
# Conditionals in iterables
# We can also define different rules for each element in the object as follows:
num3 = [10,20,30]
num4 = [i/10 if i == 20 else i-10 if i < 15 else i+5 for i in num3]
print(num4)

# Now let's return to our credit card fraud detection data set and do some examples about
# list comprehensions
# we will try to classify the transaction amounts whether they are high valued or low valued transactions

threshold1 = sum(data.Amount)/len(data.Amount)
print(threshold1)
data["Amount_level"] = ["high" if i > threshold1 else "low" for i in data.Amount]
data.loc[:10,["Amount","Amount_level"]]