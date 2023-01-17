# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def square_sum1(n):

    """

    Takes an input n and return sum of the squares of numbers from 0 to n

    

    """

    sum_ = 0

    for x in range(0,n+1):

        sum_ = sum_ + x**2

    

    return sum_
square_sum1(3)
def square_sum2(n):

    result = n*(n+1)*(2*n+1)/6

    return result
square_sum2(3)
%timeit square_sum1(3)
%timeit square_sum2(3)
def find_first_index(list1):

    print(list1[0])

    

print("First index of [1,2,3] is :")

find_first_index([1,2,3])



print("First index of [2,3,4,5,6,7,8,9,10] is :")

find_first_index([2,3,4,5,6,7,8,9,10])
def values_of_list(list1):

    for each in list1:

        print(each)



list1 = [1,2]

list2 = [3,4,5,6,7]



print("Values of list1") 

values_of_list(list1)



print("Values of list2")

values_of_list(list2)
def cubic_big_o(list1):

    for item1 in list1:

        for item2 in list1:

            for item3 in list1:

                print(item1, item2, item3)

                

cubic_big_o([1,2,3])

        
def linear_big_o(list1):

    for each in list1:

        print(each)

        

linear_big_o([1,2]) # Scale of this algorithm is O(n)
def linear_big_o_1(list1):

    for each in list1:

        print(each)

    for each in list1:

        print(each)

        

linear_big_o_1([1,2]) # Scale of this algorithm is O(2n)
def example(list1):

    print(list1[0])



example([1,2])  # scale is O(1)
def example(list1):

    print(list1[0])

    for each in list1:

        print(each)



example([1,2])  # scale is O(1+n), here is 1 is insignificant constant
n = 2

for x in range(n):

    for y in range(n):

        print("example1")
n = 2 

for x in range(n):

    print("example2")
n = 2 

for x in range(n):

    print("1")

for x in range(n):

    print("2")

for x in range(n):

    print("3")
list1 = [[1,2,3],[4,5,6]]



import copy

list2 = copy.deepcopy(list1)

print("list2:",list2)



list1[0][2] = 100

print("list1:", list1)

print("list2 after list1 changed:", list2)

import copy

family = {"father":"john", "mother":"linda", "pets":["cat"]}

new_family = copy.copy(family)



print("new_family =", new_family)



# adding new item in family

family["child"]="maria" 

print("new_family after adding new item Child in family =", new_family)



# adding new item in list of pets

family["pets"].append("dog")



print("new_family after adding item of pets in family", new_family)
import copy

old_list = [[1,2,3,4]]

new_list = copy.copy(old_list)

old_list[0][1]=100



print("old_list:", old_list)

print("new_list:", new_list)

list1=[1,2,3]

list1[1]=1000

print(list1)

tup =(1,2,3)

# tup[1]=100 # it can not be changed. 
# If we want to learn what copy() function does?



help(copy.copy)
copy.copy.__doc__
# If we want to learn what are the attributes and methods of copy() function?

dir(copy.copy)
list1 = [1,2,3,4,5,6]

print("first item of list1:", list1[0])

print("last item of list1:", list1[-1])

print("2nd item starting from end:", list1[-2])
def print_items(*args):

    for each in args:

        print(each)



print_items(1,2,3,4,5)
def print_items(**kwargs):

    for each in kwargs:

        print(each,"-->", kwargs[each])

        

print_items(a=1, b=2, c=3)
def print_items(**kwargs):

    for keys, values in kwargs.items():

        print(keys,"-->", values)

        

print_items(a=1, b=2, c=3)
list1 = [0,1,2,3,4,5,6,7,8,9,10,11,12]



from random import shuffle

shuffle(list1)  # making random

print("shuffled:", list1)

list1.sort()    # turning back to first situation

print("back to initial:", list1)
a = "HelloWorld"

joined = ".".join(a)

print("joined",joined)

splitted = joined.split(".")

print("splitted",splitted)

print("".join(splitted))
string = "    trial    "

print(string.lstrip())  # deleting leading whitespace

print(string.rstrip())  # deleting trailing whitespace

print(string.strip())   # deleting both whitespaces
a = "world"

print(a.upper())

print(a.upper().lower())
def trial():

    "Complete the definition later"

    pass
for each in range(10):

    print(each)

    if each == 5:

        break
for each in range(10):

    if each % 2 == 0:

        continue   # skips rest of the code from here

    print(each, "is odd number")

        
for each in range(10):

    if each == 2:

        continue    # when each = 2, skips rest of the code, moves on to next iteration, does not print "a"

        print("a")

    print(each)       