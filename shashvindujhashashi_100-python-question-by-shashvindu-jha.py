# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
num = float(input("Input a number: "))

if num > 0:

   print("It is positive number")

elif num == 0:

   print("It is Zero")

else:

   print("It is a negative number")


num = int(input("Enter a number: "))

if (num % 2) == 0:

   print("{0} is Even".format(num))

else:

   print("{0} is Odd".format(num))
year=int(input('year'))

if(year%4==0 and year%100!=0 or year%400==0):

    print("leap year")

else:

    print("not a leap year")
##### 4

# Python program to find the largest among three numbers



# taking input from user



num1 = float(input("Enter 1st number: "))

num2 = float(input("Enter 2nd number: "))

num3 = float(input("Enter 3rd number: "))



if (num1 >= num2) and (num1 >= num3):

   l = num1

elif (num2 >= num1) and (num2 >= num3):

   l = num2

else:

   l = num3



print("The largest number among",num1,",",num2,"and",num3,"is: ",l)
#5

print("List of months: January, February, March, April, May, June, July, August, September, October, November, December")

month_name = input("Input the name of Month: ")



if month_name == "February":

    print("No. of days: 28/29 days")

elif month_name in ("April", "June", "September", "November"):

    print("No. of days: 30 days")

elif month_name in ("January", "March", "May", "July", "August", "October", "December"):

    print("No. of days: 31 day")

else:

    print("Wrong month name") 
#6

# Python program to display all the prime numbers within an interval



lower = int(input("lower no "))

upper = int(input("upper no "))



print("Prime numbers between", lower, "and", upper, "are:")



for num in range(lower, upper + 1):

   # all prime numbers are greater than 1

   if num > 1:

       for i in range(2, num):

           if (num % i) == 0:

               break

       else :

           print(num)
# Python program to find the factorial of a number provided by the user.



# change the value for a different result



num = int(input("Enter a number: "))



factorial = 1



# check if the number is negative, positive or zero

if num < 0:

    print("Sorry, factorial does not exist for negative numbers")

elif num == 0:

    print("The factorial of 0 is 1")

else:

    for i in range(1,num + 1):

        factorial = factorial*i

    print("The factorial of",num,"is",factorial)
# To take input from the user

num = int(input("multiplication table of? "))



# Iterate 10 times from i = 1 to 10

for i in range(1, 11):

   print(num, 'x', i, '=', num*i)
# Program to display the Fibonacci sequence up to n-th term



nterms = int(input("How many terms? "))



# first two terms

n1, n2 = 0, 1

count = 0



# check if the number of terms is valid

if nterms <= 0:

   print("Please enter a positive integer")

elif nterms == 1:

   print("Fibonacci sequence upto",nterms,":")

   print(n1)

else:

   print("Fibonacci sequence:")

   while count < nterms:

       print(n1)

       nth = n1 + n2

       # update values

       n1 = n2

       n2 = nth

       count += 1
def fib(n):

    a,b=0,1

    if n==1:

        print(a)

        

    else:

        print(a)

        print(b)

        

        for i in range(2,n):

            c=a+b

            a=b

            b=c

            print(c)

fib(int(input("How many terms? ")))
num = int(input("Enter the value of n: "))

n=num

sum = 0



if num <= 0: 

    print("Enter a whole positive number!") 

else: 

    while num > 0:

        sum = sum + num

        num = num - 1;

    # displaying output

    print("Sum of first",n," natural numbers is: ", sum)
# Sum of natural numbers up to num



num =int(input("Enter the value of n: "))



if num < 0:

   print("Enter a positive number")

else:

   sum = 0

   # use while loop to iterate until zero

   while(num > 0):

       sum += num

       num -= 1

   print("The sum is", sum)