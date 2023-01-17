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
a = 7

c = 10 
a = c
a
c
type(a)
a = 't'
type(a)
a
a = 5.72242
type(a)
a + 10
flag = True
type(flag)
not not flag
greet = "Hello Everyone !"
print (greet)
greet_message = "Hello"

name = "Ram"
#String concatenation

print(greet_message + " Mr." + name)
#Operators

a = 43

b = 9
print(a + b)
print(a - b)
print(a * b)
print(a/b)
print(round(a/b))
print(a//b)
print(a%b)
print(a ** b) #43 ^ 9
print((-9) ** (0.5))
round(5.6875, 1)

#Round to nearest (10^(-x))
import math as m
m.floor(6.999)
m.ceil(7.001)
s = "a"
s * 100
for a in range(500):

    print(a)
n = 10

n = int(input('Enter'))
print(n)
#If n < 30, we say "less"

n = 30

if n < 30:

    print("less")

print("Out of if block !!")
#If n <= 30, we say "less"

#Else, we say more

n = 30

if n <= 30:

    print("less")

else:

    print("more")



print("Out of if-else block !!")
#If n <= 30, we say "less"

#Else, we say more

n = 30

if n < 30:

    print("less")

elif n > 30:

    print("more")



print("Out of if-else block !!")
#If n < 10, we say "less than ten"

#Else if n < 20, we say "less than twenty"

#Else if n < 30, we say "less than thirty"

n = 32

if n < 10:

    print("less than ten")

elif n < 20:

    print("less than twenty")

elif n < 30:

    print("less than thirty")



print("Out of if-elseif block")
#If n < 10, we say "less than ten"

#Else if n < 20, we say "less than twenty"

#Else if n < 30, we say "less than thirty"

#Else, we say "greater"

n = 32

if n < 10:

    print("less than ten")

elif n < 20:

    print("less than twenty")

elif n < 30:

    print("less than thirty")

else:

    print("greater")



print("Out of if-elseif block")
#n = 5 ==> Assign 5 to n

#n == 5 ==> check if n is equal to 5 

n = 10

if n == 5:

    print("five")

elif n == 7:

    print("seven")
#Input n from user

#If n < 1000, print "less"

#If n > 1000, print "more"

#If n = 1000, print "equal"
#Unreachable code (Dead code)

n = 10

if n < 10:

    print("less than 10")

elif n >= 10:

    print("geq to 10")

else:

    print("sample")
print("1")

print("2")

print("3")

print("4")

print("5")

print("6")

print("7")

print("8")

print("9")

print("10")
# Generic statement

#Pass the values dynamically

#range(10) -> [0,9]

#range(2, 10) -> [2,9]

#range(2, 10, 3) -> [2, 5, 8]

for i in range(10): # i in [0,1,2,3,4,5,6,7,8,9]

    print(i)

    print(i ** 2)
#i -> Control variable

for i in range(10): #i in [0,1,2,3,4,5,6,7,8,9]

    print("Hello Everyone!")
i = 0

while i < 10:

    print("Current value of i is ", i)

    print("Hello Everyone !")

    i = i + 1  #i = 1 #Update the current variable

    print("i for next iteration is ", i)
i = 4

i = i + 1

i
i - 0

for i in [2, 10, 5, 6]: #List

    print("Hello Everyone!")

    print(i)
i = 0

while i <= 5:

    print(i)

    i = i + 1

print("sample Output")
for i in range(1, 3):  #i in [1,2,3,4,5,6,7,8,9,10]

    print("current value of i is ", i)

    for j in range(1, 4): #j in [1,2,3,4,5,6,7,8,9,10]

        print("current value of j is ", j)

        print(i,' multiplied by ',j, " gives ", i * j)
i = 4

j = 5

print(i,' * ',j, " gives ", i * j)
#i = 1

#j from 1 to 10

#i = 2

#j from 1 to 10

#i = 3

#j from 1 to 10

for i in range(5): #i in [0,1,2,3,4]

    print("current value of i is ", i)

    for j in range(i): #range(x) -> [0, x-1] [range(0)-> ]

        print("j is ", j)

        print(i)
#1

#2

#3

#4

#5
#1

#22

#333

#4444

#55555
#55555

#4444

#333

#22

#1
#11111

#2222

#333

#44

#5
#5

#44

#333

#2222

#11111
#     1

#    2 2

#   3 3 3

#  4 4 4 4

# 5 5 5 5 5

#6 6 6 6 6 6