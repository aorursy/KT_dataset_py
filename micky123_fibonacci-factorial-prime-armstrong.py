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
a = 25

b = 15.20

d =  9.322e-36j

str = "this is a example of string"

print(type(a)) # type() is use to check the type of variable.

print(type(b))

print(type(d))

print(type(str))
lst = ["Abc",1,23,546,"fht"]

tpl = (1,2,4,5,"abc")

dict1 = {"a":[1,2,3,4,5], "b": [4,5,6,7,8]}

set1 = {1,2,4,("a","b")}

print(type(lst))

print(type(tpl))

print(type(dict1))

print(type(set1))
print("Hello World")
def add(a,b):

    return a+b

add(10,3)
def square_root(num):

    return num**0.5

square_root(4)
def area_of_triangle(a,b,c):

    s = (a+b+c)/2 #semiperimeter of triangle

    ar = s*(s-a)*(s-b)*(s-c)

    print("Area of triangle is:",square_root(ar))

area_of_triangle(4,4,5)
for i in range(1,10):

    print(i, end = ",")
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
for i in range(1,11):

    print(i, end = ",")

    i = i+1
for i in range(2,100,2):

    print(i, end = ",")
for i in range(1,100,2):

    print(i, end = ",")
def fib(n):

    if n == 0:

        return 0

    elif n == 1:

        return 1

    else:

        return fib(n-1) + fib(n-2)

fib(12)
upper = 100

lower = 2

def print_prime_number(number):

    for number in range(lower, upper +1):

        if number > 1:

            for i in range(2, number):

                if (number % i) == 0:

                    break

            else:

                    print(number, end = ",")

print_prime_number(100)
def check_prime_number(num):

    if num > 1:

        for i in range(2, num-1):

            if (num%i) == 0:

                print(num, "is not a prime number","it is divisible by", i)

                break

        else:

                print(num,"is a prime number")

        

check_prime_number(33)
num = 152 #you can use any number of your choice

def check_num_armstrong(num):

    sum = 0

    temp = num

    while temp > 0:

        digit = temp%10

        sum += digit**3

        temp //= 10

    if num == sum:

       print(num,"is an Armstrong number")

    else:

       print(num,"is not an Armstrong number")

check_num_armstrong(num)
num = 5 #you can use any number of your choice

def factorial(num):

    factorial = 1

    if num == 0:

        print("factorial of 0 number is 1")

    elif num < 0:

        print("Sorry Factorial of negative number is not available")

    else:

        for i in range(1, num +1):

            factorial = factorial*i

        print("Factorial of", num, "is", factorial)

factorial(num)        

    