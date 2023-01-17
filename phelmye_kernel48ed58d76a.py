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
print(11%3)
def mutiple_of_three(x):

    a = int(x)

    if a % 3 == 0:

        print("TRUE: {} is devisible by 3".format(x))

    else:

        print("WRONG! {} is indevisible by 3".format(x))

        

mutiple_of_three(21)
print("hello")
def print_multiple_of_three(a):

    if a % 3 == 0:

        print(f"{a} is a multiple of 3")

    else: 

        print(f"Sorry {a} is not a multiple of 3")

        

        

print_multiple_of_three(12)

print_multiple_of_three(7)
def multiples_of_three(x):

    if x%3 == 0:

        return True

    else:

        return False

    

multiples_of_three(7)
def print_multiples_of_three():

    num = int(input("please enter a number"))

    if num % 3 == 0:

        print("{} is a modulo".format(num))

    else:

        print("{} is not a modulo".format(num))



print_multiples_of_three()

def print_multiples_of_three(self):

    print("this program is to check multiples of three")

    num = int(input("enter a number"))

    if (num % 3 ) == 0:

        print("{} is"" a"" multiple of 3 : " .format(num) + "True")

       # print ("true")

    else:

        print("{} is "" not "" a "" multiple of 3 : " .format(num) + "False")

       # print ("False")

        

        

print_multiples_of_three(9)

 
"""

this program is meant to check if a number is even or odd

enter value and check for odd or even

"""

print("this program is meant to check if a number is even or odd")

first_number = int(input("enter your value:"))



if (first_number % 2) == 0:

  '''

    I used string formatting to display the entered value by the user

  '''

  print(" {} is Even".format(first_number))

else:

   print ( "{} is Odd".format(first_number))