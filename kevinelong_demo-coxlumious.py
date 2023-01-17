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
# define a list of numbers and get the average

data = [98,32,45,66,12]



total = 0



#Loop through each number adding it to the toal



# set the count to be the lenggth of the data

for n in data:

    total = total + n

    

count = len(data)



average = total / count



print(average)



# function generate a randome number between 1-6

# def roll(sides = 6):

#     # ...

# # call the first function n times return the total



# def roll_many(quantity= 3, sides = 6):

#     #...

    

# # call the second n time and return the occurances of each number as a dict



# def roll_a_lot(times = 100):

#     data = {}

#     #...

#     return data



# example_output = {

#     3 => 3

#     4   => 5

#     5 => 8

#     ...

#     10 => 33

#     11 => 35

#     ...

#     16 => 8

#     17 => 5

#     18 => 2

# }
from random import randint





# function generate a random number between 1-6

def roll(sides=6):

    return randint(1, sides)



# ...

# call the first function n times return the total



def roll_many(quantity=1, sides=6):

    total = 0

    for n in range(0, quantity):

        total = total + roll(sides)

    return total



# ...



# call the second n time and return the occurances of each number as a dict



def roll_a_lot(times=100):

    data = {}

    for n in range(0, times):

        r = roll_many(3)

        if r not in data:

            data[r] = 0

        data[r] = data[r] + 1

    return data



data = roll_a_lot(2000)



print(data)
# print each key in the data followed by the quantity in key order

# 3|2

# 4|5

# ...

#18|4



#  3|

#  4|

#  5|1

#  6|

#  7|2

#  8|1

#  9|1

# 10|1

# 11|1

# 12|1

# 13|2

# 14|

# 15|

# 16|

# 17|

# 18|
# PART 5

# Print a Horizontal Bar chart like this showing each roll and a star for each unit of quantity:

# 3|*****

# 4|*******

# 5|***********

# etc
# BONUS FOR PART 5

# Scale to fit sceen width 80 characters so large sets like 10,000 fit

def chart(data, minimum=3, maximum=18, limit=80):



    for n in range(minimum, maximum + 1):

        s = ""

        if n < 10:

            s = s + " "

        s = str(s) + str(n) + "|"

        if n in data:

#             stars = data[n]

            fraction = data[n] / (maximum * len(data))

            stars = int(fraction * limit)

            # print(data[n])

            for i in range(0, stars):

                s = s + "*"

        print(s)





chart(data)
