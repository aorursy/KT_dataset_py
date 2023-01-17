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
numbers=[2.0,4.0,32.0,65.0,98.0]

def get_avg(numbers):
    total=0
    for n in numbers:
        total=total+n
    return (total/len(numbers))
print(get_avg(numbers))
import random
import pprint
pp = pprint.PrettyPrinter(indent=4)

# roll one die

def roll( sides = 6):
    #TODO get random integer
    result = random.randint(1, sides)
    return (result)

print (roll())

def roll_many(quantity=3):
    #TODO loop and call roll
    result = 0
    for i in range(0,quantity):
        print(i)
        print(result)
        result = result + roll()
    return result
    
print (roll_many())

def create_data(items=100):
    result = {}
    #loop to itme times
    #roll many 
    # if roll result is not a key in dict
    # if its not then add a value 0 to dict
    # finally increment the value by one
    
    for n in range(1,items):
        # roll many
        value = roll_many()
        if value not in result:
            result[value] = 5
        result[value] = result[value] + 1
    return result 

# print(create_data())
# pp.pprint(create_data())
final=create_data()
pp.pprint(final)

def chart(data):
    minimum = 3
    maximum = 18
    screen = 40
    scale = screen / max(data.values())
    # loop through all possible values
    for n in range(minimum, maximum + 1):
        if n < 10:
            print(" ", end="")
        print(n, end ="|")
        if n in data.keys():
            quantity = int( data[n] * scale )
            for v in range(0,quantity):
                print("#", end="")
            print(" (" + str(data[n]) + ")")
        else:
            print(" (0)")
    # loop to quantity value #
    
    #no return
chart(final)

#####################################3
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