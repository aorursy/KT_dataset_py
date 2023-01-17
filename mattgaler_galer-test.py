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
#Make a list of numbers
# make a function that takes a list as a parameter and calc average

numbers = [12.1, 34.2, 65.0, 76.0]

def get_average(numbers):
    total = 0.0
    for n in numbers:
        total = total + n
    return total / len(numbers)

print(get_average(numbers))

import random
import pprint

pp = pprint.PrettyPrinter(indent=4)

# roll one die
def roll(sides = 6):
    #TODO get random integer
    result = random.randint(1, sides)
    return result

print(roll(6))

def roll_many(quantity=3):
    #TODO loop and call roll
    result = 0
    
    for n in range(0,quantity):
        result = result + roll(6)
    return result

print(roll_many())

def create_data(items=10000):
    result = {}
    
    for item in range(0, items):
        roll = roll_many(3)
        if roll not in result:
          result[roll] = 0
        result[roll] += 1
#    sorted_result = {k: result[k] for k in sorted(result)}
#    return sorted_result
    return result

final = create_data()
print(final)

pp.pprint(final)

    

#print horizontal bar chart
# 3|#### 4
# 4|###### 6
# 5|## 2
#...
#10|########### 11
#...
#18| 0

def chart(data):
    #loop through all possible values
    minimum = 3
    maximum = 18
    screenwidth = 40
    scale = screenwidth / max(data.values())
    if scale > 1:
        scale = 1
    print('SCALE = {:.2f}'.format(1/scale))
    for number in range(minimum, maximum + 1):
        print('{0:2d}'.format(number), end="|")
        if number not in data.keys():
            print(' (0)')
        else:
            scaled = int( data[number]*scale)
            for pounds in range(0,scaled):
                print("#", end="")
            print(' (' + str(data[number]) + ')')
    
    #no return

chart(final)