import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import seaborn as sns
myfirstvar = 'Hello world!' # String 'something'

myfirstvar = 'drug discovery'

anothervar = 'Hello world!'

myfirstvar = anothervar

print(myfirstvar)
myfirstnumber = 1

secondnumber = 1.0

thirdnumber = 1.
print(type(myfirstnumber))

print(type(secondnumber))
first = 9

second = 10
first + second

### Multiply
first * second
### Division

first/second
first//second
second**first # 10^9
(first+second)/second
myfirstvar
# Splitting a string - split()

myfirstvar.split(' ')
myfirstvar.split('ello')
# Remove part of the string - strip()

myfirstvar.strip('world!')
# Replace part of a string - replace()

myfirstvar.replace('world','Brazil')
myfirstvar.rstrip('!').replace('world','Brazil')
mylist = [1,2,3,4,5]

print(mylist)



# Get me the length of the list

print(len(mylist))
# Indexing

## Get first element

print(mylist[0])

print(mylist[-1])

print(mylist[-2])

print(mylist[3])
# Slicing

print(mylist[0:3]) # [first,last) < includes first, excludes last

print(mylist[-2:-1])

print(mylist[0:1])
secondlist = [1,1,1,1,1,0,0,2,0,0,0,5,4,3]
mydict = {'key':'values'}

print(type(mydict))
a = mydict['key']

b = mydict.get('key')

print(a,b)
# Example

mydict2 = {'ATOM' : ['C1','C2','N2'],

          'x_axis':[10.43,49.12,50.5],

          'y_axis' : [0.3,0.4,-.5]}
mydict2['ATOM']
myset = set(secondlist)

print(myset)

print(type(myset))
list_to_set = list(myset)

print(list_to_set)

print(type(list_to_set))
mytuple = (1,2,3,4,5)

mytuple[0:3]



#users, films = get_rating('file.txt')
mythirdlist = [[1,2,3,4],[12,10,11,14]]
print(len(mythirdlist))
mythirdlist[0][0:3]
# for loop

# for x in somewhere:
b = 1

for x in mythirdlist[0]:

   # a = 0

    print('Original b = {}'.format(b))

    b = b + x

    print('New value = {}\n'.format(b))
b
b = 1

while b < 5:

   # a = 0

    print('Original b = {}'.format(b))

    break
mythirdlist = [1,2,3,4,5,6,7,8,9,10]
for x in mythirdlist:

    if x >= 5:

        print('Adding one to {} = {}'.format(x,x+1))

        

    elif x <= 2:

        print('Adding two to {} = {}'.format(x,x+2))

        

    else:

        print('Doing nothing to {}'.format(x))



        
def myfunction(x):

    '''This function adds one'''

    if x >=5:

        return x + 1

    else:

        return 0
testing = map(myfunction,mythirdlist)
list(testing)
testing2 = [x+1 for x in mythirdlist if x >= 5]
testing2