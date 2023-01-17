# we use this data for python data science toolbox
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.

import os
print(os.listdir("../input")) # we can import data from this directory
# first we need to import data which we use
mydata = pd.read_csv('../input/column_2C_weka.csv')
# data info (name of columns, data type, memory usage)
mydata.info()
#first seven datas
mydata.head(7)
def tuble(): #  define function
    t = (1,5,7)
    return t
a,b,c = tuble()
print(a,b,c)
#SCOPE
#global: defined at main body in script
#local: defined in a function

x = 2 # global
def f():
    x = 3 # local
    return x
print(x)      # x = 2
print(f())    # x = 3
# if there is not a local scope in our functions
x = 5
def f():
    y = 2*x        # there is no local scope x
    return y
print(f())         # first this function try to find Local Scope(LS)
#it can not find LS and it uses global scope x
#How can we know what is built inside of the scope
import builtins
dir(builtins)
def square():
   #return values of the square
    def add():
        x = 2
        y = 3
        z = x + y
        return z # return for add function
    return add()**2 # return for square function
print(square()) 


def f(a, b = 1, c = 2): # if we dont change values of b and c,
    #b and c will be default arguments
    y = a + b + c
    return y
print(f(5)) # a=5 and b,c is defined in f() function as a default values

print(f(5,4,3)) # we can change default arguments
# if we want to use flexible arguments, we need to define in our
#function "*args"
def f(*args): # flexible defining
    for i in args:
        print(i)
f(1) # with this we can chose any number to use def f()
print("")
f(1,4,3,7)
# flexible arguments **kwargs (key and value argument)that is dictionary
def f(**kwargs):
    for key, value in kwargs.items():  # If you do not understand this part turn for loop part and look at dictionary in for loop
        print(key, " ", value)
f(country = "poland", capital = "warsaw", population = 1745000)
# where x is name of argument, lambda variables:function
square = lambda x: x**2     
print(square(4))
tot = lambda x,y,z: x+y+z   # where x,y,z are names of variables
print(tot(1,2,3))
#Like lambda function but it can take more than one arguments.
#map(func,seq) : applies a function to all the items in a list

number_list = [1,4,7]
y = map(lambda x:x**2,number_list)
print(list(y))
soccer = "Nakata"
iteration = iter(soccer) # iteration command
print(next(iteration))    # print next iteration
print(next(iteration)) # print next iteration
print(*iteration)         # print remaining iteration
#ZIP
list1 = [1,2,3,4]
list2 = [5,6,7,8]
z = zip(list1,list2)
print(z)
z_list = list(z)
print(z_list)

# we can make unzip our zip datas
un_zip = zip(*z_list) 
un_list1,un_list2 = list(un_zip) # unzip returns tuble
print(un_list1) # type of this will be tuple
print(type(un_list1))

un_list2 = list(un_list2) # we can change type of unzip data 
print(un_list2)
print(type(un_list2))

num1 = [1,4,7,9]
for i in num1: # we can look(see,find) inside of the num1
    print(i)
    
num2 = [i + 1 for i in num1 ]
print(num2)
# Conditional iteration
num1 = [5,10,15,4,20]
num2 = [i**2 if i == 10 else i-5 if i < 7 else i+5 for i in num1]
print(num2)
#lets return our csv and make a list comprehension example
# lets classify datas whether they have high or low speed
#Our threshold is average pelvic_incidence
threshold = sum(mydata.pelvic_incidence)/len(mydata.pelvic_incidence)
mydata["High/Low"] = ["high" if i > threshold else "low" for i in mydata.pelvic_incidence]
mydata.loc[:10,["High/Low","pelvic_incidence"]] 
