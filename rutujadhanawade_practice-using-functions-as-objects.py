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
#assign an int to variable

a=5



#Use an int as an argument in a function

def add_two(a):

    return a+2

add_two(6)



#Return a string in a function

def is_even(a):

    return 'even' if a%2==0 else 'odd'

is_even(10)



#A boolean value in tuple

(True,7,8)



a,b,c=(True,7,8)

print(a,b,c)



p,q,r=['Mumbai','Delhi','Kolkata']

print(p,q,r)
############## Funtion - Variable Assignment

def greeting(name):

    hello='Hello '+name

    return hello

print(greeting('Rutu'))



say_hello=greeting

say_hello('Simran')



#Note: greeting and hello refer to same object in memory.

print(id(greeting))

print(id(say_hello))



#If we delete greeting(),the name 'greeting' will become undefined,while the deletion doesnt affect underlying object.

#Thus important concept is that variable referring to the function is different from the actual object saved in the memory.



del greeting



say_hello('Neha')

#greeting('Neha')
################ Use Function as another Function's Argument

def combine_two_numbers(how_to, numbers):

    return how_to(numbers)

def add_two_numbers(numbers):

    a,b=numbers

    return a+b

def multiply_two_numbers(numbers):

    a,b=numbers

    return a*b



print(combine_two_numbers(add_two_numbers,(4,5)))

print(combine_two_numbers(multiply_two_numbers,(4,5)))

    
#################### Use Function as Return Value

def add_number_creator(number):

    def add_number(a):

        return a+number

    return add_number



add_three=add_number_creator(3)

add_five=add_number_creator(4)



print(add_three(20))

print(add_five(10))
###########  Function as a part of Another OPbject

add_functions=[add_number_creator(0),add_number_creator(1),add_number_creator(2)]



for i,func in enumerate(add_functions):

    a=8

    print("a is 8,adding "+str(i)+' is '+str(func(a)))