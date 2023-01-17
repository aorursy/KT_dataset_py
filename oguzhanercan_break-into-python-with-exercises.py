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
print("helllo world") #printing
var1 = 15.5

var2 = 10

var3 = var1 + var2

print("var 3 = ", var3) # You dont have to spesify the type of variable like int float char
var4 = "Hello"

var5 = "World"

var6 = var4+" "+var5

print(var6)
list1 = ["apple","banana"]

print(list1)



tuple1 = ("apple","banana")

print(tuple)



len(list1) # lenght function



list2 = [1,2]

print(list2)



list3 = list1+list2

print(list3)



dict = {

  "brand": "Ford",

  "model": "Mustang",

  "year": 1964

}

print(dict)







name = input("please enter your name :")

print("Welcome ",name)
def multiplication(a,b):

    return a*b

multiplication(5,5)
number = int(input("please enter a number : "))



if number < 0 :

    print("negative")

elif number == 0 :

    print("zero")

else : #here positive or we can write elif number > 0 :

    print("positive")
for i in range(0,4):

    print(i)
for i in list:

    print(i)


def prime(a):

    b = 1

    for i in range(2,int(a/2+1)):

        if a%i == 0 :

            b = 0

            break

    if b == 1:

        print("prime")

    else :

        print("not prime")

prime(17)
def guess_number():

    number = int(input("please enter the number player will try to find :"))

    guess = int(input("player's guess : "))

    while number != guess :

        if guess > number :

            guess = int(input("Enter smaller number ."))

        else :

            guess = int(input("Enter bigger number ."))

    print("correct")



guess_number()