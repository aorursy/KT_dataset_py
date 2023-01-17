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
3*(7+2)
z = 5/2

z
s='abc'

x="abc"

print(s==x)
height = 1.79

weight = 68.7

print("Height : ",height)

print("Weight : ",weight)
# Calculate BMI

bmi = weight/height ** 2

print(" BMI : ", bmi)
x,y=2,3

print("x,y : ",x,y)
a=[1,2,3] 

b=a  # b now references waht a references

a.append(4)

print("a,b :",a,b) # see both a and b show the same result

b.append(5)

print("a,b :",a,b) # see both a and b show the same result

x = 3

y = x

print("x,y : ", x,y)

y = 4 # This assignment created a new reference for y

print("x,y : ", x,y)
x="Hassan"

y=x

print("x,y : ", x,y)

y="Amin"

print("x,y : ", x,y)

a=''

b=''

a=[1,2,3]

b=a

print("a,b : ", a,b)

b=['a','b','c'] # Assignment creates a new reference

print("a,b : ", a,b)

a=b

b.append(4) # Append makes changes in place without creating a new reference

print("a,b : ", a,b)



# Printing types of variables

print("Type of a,b,x,y", type(a),type(b),type(x),type(y))
fam = ['liz',123,'emma', 134,'john',546]

print(fam)

print(fam[3])

print(fam[-2])

print(fam[-3])



# List Slicing

print(fam[2:4]) # 

print(fam[2:])

print(fam[:4])



fam = ['liz',123,'emma', 134,'john',546]

print(fam)

# Changing list elements

fam[-1] = 777

print(fam)

fam[0:2]=['fiz',444]

print(fam)

fam = ['liz',123,'emma', 134,'john',546]

print(fam)

fam_ext = fam + ['nano',666]

print(fam_ext)

del(fam_ext[-1])

print(fam_ext)
def my_function():

  print("Hello from a function")



my_function()

def my_function(fname):

  print(fname + " Refsnes")



my_function("Emil")

my_function("Tobias")

my_function("Linus")

def my_function(country = "Norway"):

  print("I am from " + country)



my_function("Sweden")

my_function("India")

my_function()

my_function("Brazil")

def my_function(x):

  return 5 * x



print(my_function(3))

print(my_function(5))

print(my_function(9))
