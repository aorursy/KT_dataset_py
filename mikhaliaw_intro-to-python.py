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
3+5
var1 = 35

var1
3 **2
my_str="String1"

my_str_2= 'String 2'

my_str_3 ="""String 3"""
# we use the type function 



type(my_str)
mylist = [1,4,"Wonderful", "This is not an array !"]

mylist 
help(mylist)
mylist[2]+ " "+ mylist [3]
tup1 = (5,6,4)
tup1 [0]
person = {

    "name":"Mikhalia",

    "age":22

}
person ["name"]
# add a new key 

person["shirt_coloyr"] = "purple"

person 
#change value to existing property 

person[ "shirt_colour"]="black and white"

person 
persons = [

    {

    "name":"Mikhalia",

    "age":22

}, 

    {

    "name":"Shanice",

    "age":21

}, 

    {

    "name":"Rashell",

    "age":23

}

]

persons

persons [1]
persons[1]["name"]
persons [1]["age"] = 45
persons[1]["age"]
if 5 == 3 and True:

    print("5 is equal to 3")

else:

    print("5 is not equal to 3")

    
for i in range (0,10):

    print(i)
for i in range(0,5,2): #increase by 2

    print(i)

    
# Using loops and conditional statements 

# Chnage update all person dictionary items in persons 

# if their age is over 30, their shirt colour should be red, otherwise blue



for i in range(0, len(persons)):

    person=persons[i]

    print("----")



    if person["age"] > 30:

        print(person,"shirt colour is red")

    else:

        print(person,"shirt colour is blue")
len(persons)
def add(num1,num2):

    return num1+num2
add(21,3)

def add3(num1,num2,num3=7):

    print("Adding --> ",num1,num2,num3)

    return num1+num2+num3

print("5 + 6? = ",add3(5,6))

print("1 + 4+5 =", add3(num3=4,num2=1,num1=5))





subtract3 = lambda x : x - 3

subtract = lambda x,y: x- y 

print( subtract3(4),

      subtract(4,6) )
 

"True" if 3 > 5 else "False"
