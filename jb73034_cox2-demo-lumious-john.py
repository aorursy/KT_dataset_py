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
# Make a list of number
# make a function that takes a list as a parameter
numbers = [12.0, 34.0, 65.0, 76.0]

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
    result = random.randint(1,sides)
    return result

#print(roll())

def roll_many(quantity=3):
    result = 0
    for i in range(quantity):
        die = roll()
        result = result + die
    return result

#print(roll_many())

def create_data(item=100):
    result={}
    for i in range(0,item):
        sum = roll_many()
        if sum not in result:
            result[sum] = 0
        result[sum]= result[sum]+1
    return result

final = create_data()
#print (final)
print (len(final))
pp.pprint(final)



def chart(data,screen = 40 ,minimum =3, maximum =18):
    scale = screen / max(data.values())
    for i in range(minimum, maximum + 1):
        print('{0:02d}'.format(i),end=' |')
        if i in data.keys():
            if scale < 1:
                quantity = int( data[i] * scale )
            else:
                quantity = data[i]
            for v in range(0,quantity):
                print("#", end="")
            print(" - " + str(data[i]) )
        else:
            print(" - 0")
            
chart(final)


40/10