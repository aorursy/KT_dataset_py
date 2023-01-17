# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Power Operator
print(2**4)
#Modulus Operator - Returns the remainder
15 % 4
#Use single quote or double quotes
#You can wrap single quotes around double quotes
text = "I can't kill a lamb"
print(text)
Name = 'Deepali'
Age = 7
print("My name is {} and I'm {} years old".format(Name, Age) )
s = 'Kaggle Competition'
print(s[0])
print(s[-1])
print(s[:5])
print(s[:-4])
print(s[-2:])
print(s[0:3])
my_list = [1, 2, 3]
print(my_list)
my_list.append(4)
print(my_list)
print(my_list[:2],my_list[1:],my_list[0:2])
my_dic = {'key1':'value1', 'key2':34 , 'key3':'Python', 'key4':[2,3,4], 'key5':['A','B']}
print(my_dic)
#Dictionaries don't maintain order
#So my_dic[1] will give an error
#To get a value, you need to access the dictionary by its key
print(my_dic['key2'], my_dic['key5'])
#Declaring a set
s = {1,2,3}
print(s)
#Sets only contain unique elements
s = {1,1,2,3,4,3,2,1,3,1}
print(s)

set (s)
#To get unique elements from a list, set function can be used
set([1,1,1,1,2,3,4,4,4,4,6,6,6])
set(['a','a','b'])
#To add elements to a set, use add method
s = {1,2}
s.add(3)
print(s)
x = 10 # Change the value of x to see what happens
if x < 5 :
    print('Less than 5')
elif (x > 5) and (x<15) :
    print('Between 10 and 15')
else:
    print('Greater than 15')
#for loop - Method1
seq = [1,2,3,4,5]
for i in range(len(seq)):
    print(seq[i])     
print(len(seq))
list(range(5))
print(len(seq))
print(range(5))
#for loop - Method2
seq = [1,2,3,4,5]
for item in seq:
    print(item)
#While Loop
i = 1
while i < 5:
    print(i*2)
    i = i + 1
#Suppose you want to create a list of table of 5
#Method 1 - Use for loop
Table = []
for i in range(10):
    Table.append((i+1)*5)
print(Table)
#Method 2 -  Use list comprehension
Table = [(i+1)*5 for i in range(10)]
print(Table)
def my_double(x):
    y = x * 2
    return y

print(my_double(5))
seq = [1,2,3,4,5]
#Suppose we want to  apply the function my_double to every element in list seq
#Method 1 - Use a for loop & keep appending the return value to other list
#Method 2 -  Use Map
map(my_double, seq) # This creates a map object
# to see the list, apply list on the map object
list(map(my_double, seq))
t = lambda x : x*2
print(t(5))
list(map(lambda x: x* 2, seq))
#Filters out elements satisfying a certain criteria
list(filter(lambda number : number % 2 == 0 , seq))
Name.capitalize()
my_dic.keys()
my_dic.values()
my_dic.items()
7**4
s = "Hi there Sam!"
print(s.split())














