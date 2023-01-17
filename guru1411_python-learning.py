# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Printing a word in Python
print ("hello")
# Printing a sentence in Python
print("Hello I am a data scientist")
# Assigning an Integer Variable - Method 1
int_variable = 10
# Assigning an Integer Variable - Method 2
intVariable = 10
# Printing the value of Integer Variable
print(int_variable)
# Getting the data type of the variable
type(int_variable)
# Assigning a float Variable
float_variable = 0.5
# Printing the value of Float Variable
print(float_variable)
# Getting the data type of the variable
type(float_variable)
# Assigning a string Variable
string_variable = "Hello"
# Printing the value of String Variable
print(string_variable)
# Getting the type of variable
type(string_variable)
# Assigning a boolean variable
boolean_variable = 10 > 20
# Printing the value of boolean variable
print(boolean_variable)
# Getting the type of variable
type(boolean_variable)
# Assigning a list variable
list_variable = [10, 30, 40, 10]
# Printing the values of list variable
print(list_variable)
# Appending a new value to a list variable
list_variable.append(60)
# Printing the values of list variable
print(list_variable)
# Getting the first value in the list variable
list_variable[0]
# Getting the second value in the list variable
list_variable[1]
# Changing the third value in the list variable from '10' to '50'
list_variable[3] = 50
# Printing the values in the list variable
print(list_variable)
# Assigning a tuple variable
tuple_variable = (10, 30, 40, 10, 60)
# Printing the values of a tuple variable
print(tuple_variable)
# Getting the type of variable
type(tuple_variable)
# Attempting to change a value in the tuple variable will throw error
tuple_variable[3] = 50

#list_variable[3] = 50
#Tuple is immutable
#List is mutable
# Assigning a set variable
set_variable = {10, 30, 40, 10, 60}
# Knowing the difference between set, list and tuple representation
print(set_variable)
print(list_variable)
print(tuple_variable)

#Set will not allow duplicates, insertion order is not preserved and does not support indexing
# Attempting to get first value in set variable like list and tuple will throw error
set_variable[0]
list_variable.append
list_variable.clear
list_variable.copy
list_variable.count
list_variable.extend
list_variable.index
list_variable.insert
list_variable.pop
list_variable.remove
list_variable.reverse
list_variable.sort
tuple_variable.count
tuple_variable.index
set_variable.add
set_variable.clear
set_variable.copy
set_variable.difference
set_variable.difference_update
set_variable.discard
set_variable.intersection
set_variable.intersection_update
set_variable.isdisjoint
set_variable.issubset
set_variable.issuperset
set_variable.pop
set_variable.remove
set_variable.symmetric_difference
set_variable.symmetric_difference_update
set_variable.union
set_variable.update
