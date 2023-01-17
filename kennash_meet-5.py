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
#loops repeatedly execute certain blocks of code

#This is great for something like lists, if you want to do something to each object in the list without writing code for each one



#Thinkback to our first lesson where we created multiple blocks of code at first, but then condensed it into one thing: computational thinking



names = ["Jeff", "Sally", "Ken", "George", "Tim", "Aaron"]

for name in names: #The name can be anything

    print(name)

    

for asteroid in names:

    print(name)

    

#for loops need: 1. variable name to use and 2. what set of values you're going to loop over



#for characters in a string

string = "elephant"

for char in string:

    print(char*2)

    

#if you just want to repeat something a certain amount of times, use RANGE



for i in range(10):

    print(i**2)

    

#Similar syntax to list slicing

for r in range(1, 10, 2):

    print(r)

    

#While loops do something WHILE something is true



i = 0

while i < 5:

    print(i)

    i = i + 1

    

#adding things to lists

end_list = []

def check_for_five(five_list):

    for num in five_list:

        if num % 5 == 0:

            end_list.append(num)

    return(end_list)



check_for_five([1, 4, 10, 15, 500])







ol = []

el = []

def check(li):

    for num in li:

        if num % 2 == 0:

            el.append(num)

        else:

            ol.append(num)

    return ol, el



check([1, 2, 3, 4, 5, 6, 7, 8])













print(log(4, 2))



import math as mt

print(mt.log(4, 2)) #dot 



print(dir(mt)) #These are all methods and objects you can call from the module



print(mt.pi)
