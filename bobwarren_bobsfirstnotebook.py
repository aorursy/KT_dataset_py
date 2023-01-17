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

import random

import pprint

# make list of numbers

# make a function that takes a list

numbers = [12.0, 34.0, 65.0, 76.0]



def get_average(numbers):

    total = 0.0

    for n in numbers:

        total = total + n

    return total / len(numbers)



print(get_average(numbers))

pp=pprint.PrettyPrinter(indent=4)



# create dictionary of random numbers between 3 and 18

# roll 1 die

def roll(sides = 6):

    #TODO get random_integer

    result = random.randint(1,sides)

    return result



print(roll())



def roll_many(quantity=3):

    #TODO loop and call roll and add 3 results

    result=0

    for n in range(0,quantity):

        result = result + roll()

    return result



print(roll_many())



def create_data(items=100):

    result = {}

    #loop to item times

    for n in range(0, items):

        value = roll_many()

        if value not in result:

            result[value] = 0

        result[value] = result[value] + 1   

    return result    

    #roll many

    # if roll result is not a key in dictionary

    #if its not then add a value of 0 to dictionary

    #finally increment the value by 1



final = create_data()

#print(final)

pp.pprint(final)    
print(final)

#Chart Horizontal Bar

#3 

def chart(data):

    #data is a dictionary keyvalue pairs

    #loop through all possible values

    

    minimum=3

    maximum=18

    screen=40

    biggest=max(data.values())

    scale=screen/biggest

    for n in range(minimum, maximum + 1):

        if n < 10:

            print(" ", end="|")

        if n in data.keys():

            quantity=int(data[n]*scale)

            for v in range(0,quantity):

                print("#", end="")

            print("  (" + str(data[n]) + ")")    

        #if n is a key in the the dictionary 

        if n in data.keys():

            for v in range(0,data[n]):

                print('#', end="")

            print(data[n])

        else:

            print(" (0)")

        

chart(final)        