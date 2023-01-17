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
# Example of User Defined Functions 



def hi(name):

    """User Defined Functions"""  #docstring of function

    print("Welcome ," , name)

hi("Meryem")
# Find the Perfect Number Between 1 and 10000 With User Defined Functions 



def perfect_num(num):



    sum = 0



    for x in range(1,num):



        if (num % x == 0):

            sum +=x

            x +=1

        else:

            x+=1

    if (sum == num):

        return True



for num in range(1,10001):

    if (perfect_num(num)):

        print(num)

# Example of Scope 



x = 5           # Global Scope (defined in main body)

print(x)



def funct():

    y = 6        # Local Scope (defined in function)

    x = 3        # Cahnge the global scope in function 

    print(x+y)

funct()
# Use the Global Scope in Function 



i = 50



def funct():

    

    global i    # Get the global scope with global method

    if i % 3 == 0 :

        return True

    else:

        return False

funct()

    

    

    
# Built in Scope



import builtins

dir(builtins)
# Example of Nested Function 



def avg():

    """Nested Function"""

    def sum():

        x = 15 

        y = 25

        z = 35

        i  =  x + y + z

        return i

    return sum() / 3

print(avg())
# Example of Default Arguments 



def info(name = 'Meryem' , surname = 'Karakas' , no = 'Null'):  # name , surname and no are default argumnets

    print("Name :", name ,"\n","Surname : " , surname , "\n","No :", no)

info()

# Change the Default Arguments 



info(no = "456787654")

# Example of Flexible Arguments



def funct(*args):  # args can be one or more

    mul = 1 

    for i in args:

        mul *= i

    return mul

print(funct(5,6,7,8,9))

        

    
# Example of kwargs



# It uses for dictionary . 



def dict(**kwargs):

    

    for i in kwargs.values(): # It can be kwargs.keys or kwargs.items

        print(i)

dict(Dostoyevski = 'Crime and Punishment', Tolstoy = "War and Peace" , Hugo = 'Les Misérables')
# Example of Lambda Function 



div = lambda x , y : x / y

print(div(54,2))

# Example of Anonymous Function 



lst = [3,5,7,9,11]

mul_list = map(lambda x : x*5 , lst)

print(list(mul_list))
# Example of Iterators 



name = "Meryem"

iterable = iter(name)

print(next(iterable))    # iterator produces next value 

print(next(iterable))

print(*iterable)
# zip() method



lst1 = [5,15,25,35]

lst2 = [10,20,30,40]



lst3 = list(zip(lst1,lst2))



print(lst3)
# unzip 



unzip = zip(*lst3)

unlst1,unlst2 = list(unzip)

print(list(unlst1) , '\n',list(unlst2))  # We convert to list because unzip returns tuple
# Example of List Comprehension 



list1 = [10,20,30]

list2 = [i*2 for i in list1]

print(list2)
# Conditionals 



list1 = [21,34,25,67,80,14]

list2 = [x if x % 5 == 0 else 0 for x in list1]

print(list2)