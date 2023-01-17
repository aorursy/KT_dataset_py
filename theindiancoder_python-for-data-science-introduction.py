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
import keyword



print(keyword.kwlist)
print(len(keyword.kwlist))
# 12_variable=10
# global=10
# This is a single line comment
'''This is a 

multi-line

comment'''
print('''This is how it would 

look when it is 

printed out''')
print("""We can use both single quotes 

and double quotes for these""")
for character in '12345':

    print('This is the {} indented line'.format(character))

print('The loop ends here')
a=1
a=1+2\

+3
print(a)
b=(1+2+

  3)
print(b)
a=10

b=5.5

c='abc'
# multiple assignment

d,e=10,12.3

f=g=20
x=3

print(id(x))

# This will print the memory location where 3 is stored and where x is pointing at
y=3

print(id(y))

# This will print the memory location where 3 is stored and where y is pointing at
y=2

print(id(y))
a=5

b=2.3

c=1+7j
print(type(a))

# a is an instance of class int
print(type(b))

# b is an instance of class float
isinstance(c, complex)

# This checks if c is an instance of class complex
x=True

print(type(x))

# x is an instance of class bool
x='string one'

y="string two"



print(type(x))

print(type(y))
z='''String 

three'''

print(type(z))
print(x[0]) # prints the first character of the string

print(x[-1]) # prints the last character of the string
s='This is python tutorial'

print(s[5:]) # prints everything from character at 5th index (starting 0) until the end

print(s[5:14]) # prints everything from character at 5th index until the 13th character (14th character is not included)
a = [20, 20.5, 'hello']

print(a[1])
print(type(a))
print(a)

print('assigning 2nd element to 100')

a[1]=100

print(a)
t=(1, 3.2, 'hello')

print(t)

print(type(t))
t[1]
# t[1]=100
s={10, 20, 30, 40, 50}

print(s)
print(type(s))

# s is an instance of class set
set([10, 20, 20, 30])

# automatically prints out unique set of items
# print(s[1])



# we can't print out particular items since it is an unordered collection of items
d={'a':'apple','b':'bat'}

print(type(d))



# d is an instance of class dict
print(d['a'])

print(d['b'])
# print(d['c'])
print(float(10))
print(int(100.23))
print(str(100))

str(100)



# quotes r 4 us
user='arun'

loc=50

print('Congratulation, '+user+ '! You completed '+str(loc)+' lines of code!')

a=[1,2, 2, 2, 3, 4, 5]

print(type(a))
s=set(a)

print(s)

print(type(s))
list(str(12345))
print('Hello World')

a=10

print('The value of a is', a)

print('The value of a is '+ str(a))
a=10;b=20

print('The value of a is {} and b is {}'.format(a,b))
# Whatever is in format(x, y, z)

# x has an index of 0

# y has an index of 1

# z has an index of 2

print('The value of b is {1} and a is {0}'.format(a,b))



# position of variables inside format is not important when we are using indexes
# we can also assign name to the variables inside format and then use them

print('Hello, {name}! {greetings}!'.format(name='Arun', greetings='Good Morning'))



# You can also think of this as key-value pairs
print('The story of {0}, {1} and {other}'.format('A','B',other='C'))
# num=input('Enter a number: ')

num='10'

print(num)

print(type(num))
x,y= 10,20



# addition

print('addition')

print(x+10)



# subtraction

print('subtraction')

print(x-y)



# multiplication

print('multiplication')

print(x*y)



# division

print('division')

print(x/y)



# modulo division - gives the remainder

print('modulo division: 15%2')

print(15%2)



# floor operator

print('floor operator: -15//2')

print(-15//2)



print('floor operator: 15//2')

print(15//2)



# exponent

print('exponent: 2 raised to the power 5')

print(2**5)
a,b=10,20



print(a<b)
print(a!=b)
a,b=True, False

print(a and b)
print(a or b)
print(not b)
a,b=10,4



# in binary:

# 10: 1010

#  4: 0100
print(a & b)

# 0000

# ((2**3)*0)+((2**2)*0)+((2**1)*0)+((2**0)*0)
print(a | b)

# 1110

# ((2**3)*1)+((2**2)*1)+((2**1)*1)+((2**0)*0)
print(~b)
x=10

print(x)
x+=10 # x=x+10

print(x)
x-=10 # x=x-10

print(x)
x*=10 # x=x*10

print(x)
a=5

b=5

print(a is b)

print(b is a)
print(id(5))

print(id(a))

print(id(b))
a=[1,2,3]

b=[1,2,3]

print(id(a))

print(id(b))
print(a is b)
print(id(a[1]))

print(id(b[1]))

x='abc'

y='abc'

print(id(x))

print(id(y))
print(x is y)
print(x is not y)
lst=[1,2,3,4]

print(1 in lst)
print('s' in 'str')
d={1:'a',2:'b'}

print(1 in d)
print('a' in d) # we can only check in the keys
print(1 in {1,2,3})
num=10

if num>10:

    print('number is greater than 10')

print('this will always print')
num=10

if num==10:

    print('number is greater than 10')

print('this will always print')
num=10

if num-10:

    # if 0: 0 is taken as False

    # everything else is taken as True: -1 2

    # except None

    print('number is greater than 10')

print('this will always print')
if None:

    # None is absence of anything

    print('number is greater than 10')

print('this will always print')    
num=10

if num>0:

    print('num is positive')

else:

    print('num is negative')
num=10

if num>0:

    print('positive number')

elif num==0:

    print('zero')

else:

    print('negative number')
num=0

if num>=0:

    print('....entering nested if else')

    if num>0:

        print('positive number')

    else:

        print('zero')

else:

    print("didn't enter nested if else")

    print('negative number')
#### Python program to find the largest among 3 numbers

num1=20

num2=20

num3=20

if num1>=num2 and num1>=num3:

    print('num1 is the largest')

elif num2>=num3 and num2>=num1:

    print('num2 is the largest')

else:

    print('num3 is the largest')

# Product of all the numbers in a list

lst=[10, 20, 30, 40, 50]

product=1

index=0

while index<len(lst):

    product*=lst[index]

    index+=1

print(product)
lst=[10, 20, 30, 40, 50]

index=0

while index<len(lst):

    print(lst[index])

    index+=1 # this part is very important in case of while loop

else:

    print('no items are left in the list')
num=int(input('Enter a number: '))



is_divisible=False



i=2

while i<num:

    if num%i==0:

        print('{} is divisible by {}'.format(num,i))

        is_divisible=True

    i+=1



if is_divisible:

    print('The number you entered: {} is not prime'.format(num))

else:

    print('The number you entered: {} is prime'.format(num))
# # syntax

# for element in sequence:

#     body of for loop
lst=[10, 20, 30, 40, 50]



product=1

for item in lst:

    product*=item

print('Product of numbers in the list: {}'.format(product))
for i in range(10):

    print(i)

    # range 10 is not creating a list

    # everytime I come back, it generates the next item in the list
range(10)
for i in range(1,20,2):

    print(i)
lst=['one','two','three','four','five']

# range(len(lst))

for index in range(len(lst)):

    print(lst[index].upper())
index_1=20

index_2=50

prime=[]



for num in range(index_1,index_2):

    if num>1:

        is_div=False

        for div in range(2,num):

            if num%div==0:

                is_div=True

        if not is_div:

            prime.append(num)

        

print(prime)

        
numbers=[1,2,3,4]

for num in numbers:

    if num==4:

        break

    print(num)

else:

    print('in the else block')

print('Outside of for loop')
num=int(input('Enter a number: '))

is_div=False

for i in range(2,num):

    if num%i==0:

        is_div=True

        break

if is_div:

    print('The number is not prime')

else:

    print('The number is prime')
lst=[]

numbers=[1,2,3,4,5]

for num in numbers:

    if num%2==0:

        continue

    lst.append(num)

else:

    print('else-block')

print(lst)