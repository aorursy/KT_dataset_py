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
# Define a list of numbers and get the average

data = [98,32,45,66,12]



total = 0



# Loop through each number adding it to the total

for item in data:

    total += item



# Set the count to be the length of the data

count = len(data)



average = total / count



# Print average

print(average)
# Using statistics library

from statistics import mean

print(mean(data))
import random

# funcation generage a ranom number between 1-6

def roll(sides = 6):

    return random.randint(1, sides)



# call the first function n times return the total

def roll_many(quantity= 3, sides= 6):

    count = []

    counter = quantity

    while counter > 0:

        count.append(roll())

        counter -= 1

    

    return sum(count)



# call the second function n times and return the occurances of each number as a dict

def roll_a_lot(times = 100):

    data = {}

    counter = times

    while counter > 0:

        key = roll_many()

        if key not in data:

            data[key] = 0

            counter -= 1

        data[key] = data[key] + 1 

        counter -= 1

    return data



data = roll_a_lot(1000)



print(data)

# print each key in the data followed by teh quantity in key order

index_data = data.keys()

for index in range(3, 19):

    if index in index_data:

        print(f'{index} | {data[index]}')

    else:

        print(f'{index} | ')    
# Replace results with * counts



index_data = data.keys()

for index in range(3, 19):

    if index in index_data:

        banner = "*" * data[index]

        if index < 10:

            print(f'  {index}| {banner}')

        else:

            print(f' {index}| {banner}')

    else:

        print(f'{index} | ')  