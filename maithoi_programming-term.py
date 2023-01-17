# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#first class function(in term of programming)

def square(n):

    return n*n



#write a map function that has a passing argument is a defined function

def map_func(func, arg_list):

    result = []

    for i in arg_list:

        result.append(func(i))

    return result



square = map_func(square, [1,2,3,4])

print('result of map_func:', square)
#Closure

def outer_func():

    name = 'mai-thoi'

    #iner function has no argument

    def inner_func():

        print('name:', name)

    return inner_func



#assign my_func variable to outer_func function

my_func = outer_func()



#print out the name of inner function

print('my_func.__name__:', my_func.__name__)



#print the outer_func function

print('outer_func():', outer_func())

print('my_func:', my_func)



#TODO

#why my_func = outer_func() but the address of these two variables are different.



#print('my_func():', my_func())

my_func()

        
#Closure in Practice

import logging

logging.basicConfig(filename='ex.log', level=logging.INFO)



#define a closure function

def logger(func):

    def log_func(*args):

        logging.info('Running "{}" with arguments {}'.format(func.__name__, args))

        print('func(*args):', func(*args))

    return log_func



#define add function

def add(x, y):

    return x+y



def sub(x, y):

    return x-y



#assign variable two defined function:

add_logger = logger(add)

sub_logger = logger(sub)



#this is Why Closure is useful. 

add_logger(2, 4)

add_logger(3,6)



sub_logger(1, 4)

sub_logger(4,1)



!cat ex.log


#Immutable and Mutable



# Immutable

a = 'mai'

print('address of a before: {}'.format(id(a)))



a = 'thoi'

print('address of a after: {}'.format(id(a)))

#String is immutable because it creates new object everytime we change values of a



#Mutable

b = ['mai', 'thoi', 'is', 'awesome']

print(id(b))

b.append('!')

print(id(b))

Name = ['mai', 'van', 'thoi']

print('Name type:', type(Name))

str_name = ''

for name in Name:

#     print(name)

    str_name += name

    print(str_name)

    print('str_name address: {}'.format(id(str_name)))

print(type(str_name))



#everytime we assign new elements to str_name, we would create new object, and memory does not like it.

#that's why Immutable and Mutable are important to know.



    
