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
3**2
my_str="String 1"

my_str_2 = 'String 2'

my_str_3 = """

   String 3

"""
# We use the type function 

type(my_str)
mylist = [1,4,"Wonderful", "This is not an array1"]

mylist
mylist[2]+" "+mylist[3]
tup1 = (5,6,4)
tup1[0]
person = {

    "name":"Gordon",

    "age":2627

}
person
person["name"]
# Add a new key

person["shirt_colour"]="purple"

person
# change value to existing property

person["shirt_colour"]="black and white"

person
persons = [

    {

    "name":"Gordon",

    "age":2627

},

    {

    "name":"Mary",

    "age":2627

},

    {

    "name":"Sue",

    "age":2627

}

]





persons
print(persons[1]["name"])

persons[1]['age']=21

print(persons)
if 5 == 3 and True:

    print("5 is equal to 3")

    print("4")

else:

    print("5 is not equal to 3")
for i in range(0,5): 

    print(i)
for i in range(0,5,2): #increase by 2

    print(i)
# Using loops and conditional statements

# Change update all person dictionary items 

# in persons

# if their age is over 30, their shirt colour should

# be red otherwise, blue
for person in persons:

    print("----")

    #print(person)

    if person["age"] > 30:

        print(person,"Over 30")

        person["shirt_colour"]="red"

    else:

        print(person,"age under 30")

        person["shirt_colour"]="blue"
persons
for i in range(0, len(persons)):

    person=persons[i]

    print("----")

    #print(person)

    if int(person["age"]) > 30:

        print(person,"Over 30")

    else:

        print(person,"age under 30")
len(persons)
type(3)
type(int("3"))
type(str(3))
def add(num1,num2):

    return num1+num2
add(5,6)
def add3(num1,num2,num3=9):

    print("Adding --> ",num1,num2,num3)

    return num1+num2+num3

print("5 + 6? = ",add3(5,6))

print("1 + 4+5 =", add3(num3=4,num2=1,num1=5))