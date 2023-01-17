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
# Make a list of numbers
# make a function that takes a list as a parameter

numbers = [12.0, 34.0, 65.0, 76.0]

def get_average(numbers):
    total = 0.0
    for n in numbers:
        total = total + n
    return total / len(numbers)

print(get_average(numbers))




import random

# roll one die
def roll(sides = 6):
    #TODO get random integer
    result = random.randint(1, sides)
    return result

print(roll())

def roll_many(quantity = 3):
    #TODO loop and call roll
    total = 0
    
    for n in range(0,quantity):
        total = total + roll()
        
    return total

print(roll_many())

def create_data(items=10000):
    #loop to item times
    # if roll result is not a key in dict
    # if its not then add a value 0 to dict
    # finally increment the value by one

    result = {}
    
    for n in range(0,items):
        #roll many
        value = roll_many()
        if value not in result:
            result[value] =0
        result[value] = result[value] + 1

    return result

# pp(final)

# example_output = {
#     3 => 2,
#     4 => 5,
#     #...
#     9 => 23,
#     10 => 18,
#     #...
#     17 => 4,
#     18 => 1
# }
final = create_data()
print(final)


import pprint
pp = pprint.PrettyPrinter(indent=4)
# pp.pprint(final)

# Chart Horizontal Bar

# 3|#### 4
# 4|###### 6
# ...
#10|########### 11
#11|######### 8 
# ...
# 17|## 2
# 18| 0

# COMMENT HIGHLIGHTED is CRTL-/

# def my_max(series):
#     biggest = series[0]
#     for n in series:
#         if n > biggest:
#             biggest = n
#     return biggest 


def chart(data):
    
    minimum = 3
    maximum = 18
    
    screen = 40
    
    biggest = max(data.values())
    scale = screen / biggest
    print("SCREEN (WIDTH in CHARACTERS): " + str(screen))
    print("BIGGEST: " + str(biggest))
    print("SCALE (SCREEN/BIGGEST):" + str(scale))
    
    # loop through all possible values
    for n in range(minimum, maximum + 1):
        if n < 10:
            print(" ", end="")
        print(n, end ="|")
        if n in data.keys():
            if scale < 1:
                quantity = int( data[n] * scale )
            else:
                quantity = data[n]
                
            for v in range(0,quantity):
                print("#", end="")
            print(" (" + str(data[n]) + ")")
        else:
            print(" (0)")
    # loop to quantity value #
    
    #no return
chart(final)
print(final.values())
#extra credit change num,ber of itesm to 10000 and have the chart scale to fit on the screen