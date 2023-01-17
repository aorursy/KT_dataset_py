import sys

import os

import time

import math
sys.version
sys.platform
os.getcwd()
os.listdir()
time.strftime("%d-%B-%Y %H:%M:%S %p",time.localtime())
2 + 3
2.5 + 6.5
10 * 3
50 / 6
50 // 6
50 % 6
4 ** 3
math.pi
math.sqrt(3 ** 2 + 4 ** 2)
math.pow(9, 1/2)
math.pow(5,3)
math.log10(100)
math.log2(32)
math.log(50)
math.e
print('Hello World')
math.e * 3
fname = 'Johnny'

lname = 'Depp'
fname
lname
print(fname, lname)
len(fname)
len(lname)
fname = 'Michael'
print(fname, lname)
iphone_purchased = 3

iphone_unit_price = 749.00
total_price = iphone_purchased * iphone_unit_price
total_price
str = 'United States'
len(str)
str[0]
str[-1]
str[0:5]
str[0:6]
str[0:4]
str[7:]
str[-6:]
str[::-1]
str[::2]
str * 2
str + ' of America'
str.capitalize()
str.lower()
str.upper()
s = str.split()

s
type(s)
s[0]
s[1]
str = 'Be the change you want to see in the world'
len(str)
for word in str.split():

  print(word)
str_split = str.split(sep=None)
str_split
str.isdigit()
str.isalpha()
str.isalnum()
str.isnumeric()
str.isprintable()
str.isupper()
str.islower()
str.count('the')
str.count('be')
str.find('an')
str.find('Be')
str_copy = str
str_copy
del str, str_copy