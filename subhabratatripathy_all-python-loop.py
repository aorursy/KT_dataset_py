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
coordinates = [(4, 5), (2, 3), (1, 2)]

print(coordinates[1])


# Function
def say_hi():
    #creating the function by 'def'
    print("Hello user")
say_hi()    #calling the function

def say_hi():
    print("Hello user")
print("1st")    
say_hi()
print("2nd")
def say_hi(name, age):
    #name and age are parameter , we are passing the parameter through the function
    print("Hello " + name + ", you are " + age)
say_hi("ram", "90")
say_hi("shyam", "67")
# Return Satement
def cube(num):
    return num*num*num
    
result = cube(4)  # CREATING A VARIABLE  
print(result)
#If satement
is_male = True
is_tall = True
if is_male or is_tall:
    print("You are a Male ")
else:
    print("You are not a Male")
#If satement
is_male = True
is_tall = False
if is_male or is_tall:
    print("You are a Male or tall or both ")
else:
    print("You are neither Male nor tall")
#If satement
is_male = False
is_tall = False
if is_male or is_tall: # in OR satement one condition have to be true
    print("You are a Male or tall or both ")
else:
    print("You are neither Male nor tall")
#If satement
is_male = True
is_tall = False
if is_male and is_tall: # in AND satement both condition have to be true
    print("You are a Male or tall or both ")
else:
    print("You are neither Male nor tall")
#If satement
is_male = True
is_tall = True
if is_male and is_tall: # in AND satement both condition have to be true
    print("You are a tall male")
else:
    print("You are neither Male nor tall")
#else if statement 
is_male = True
is_tall = False
if is_male and is_tall:
    print("You are a tall male")
elif is_male and not(is_tall):
    print("You are a short male")
elif not(is_male) and is_tall:
    print("You are not a male but tall")    
else:
    print("You are neither Male nor tall")
# max number finder
def max_num(n1, n2, n3):
    if n1 >= n2 and n1 >= n3:
        return n1
    elif n2 >= n1 and n2 >= n3:
        return n2
    else:
        return n3
    
print(max_num(300, 30, 2))   
# Builing a calculator using python
n1 = float(input("Enter the first number: "))
n1 = input("Enter the operator: ")
n2 = float(input("Enter the second number: "))





























