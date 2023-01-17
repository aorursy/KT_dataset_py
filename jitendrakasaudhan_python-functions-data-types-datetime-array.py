def add_nums(x, y, z=None, flag=False):

    if(flag):

        print('Flag is true')

    if (z == None):

        return x+y

    else:

        return x+y+z



add_nums(2,4)

add_nums(2,4,6, flag=True)
add = add_nums

print(2,3)
print(add(2,4))
type('this is string')
type(None)
type(add)
# Tuple are immutable data structures

tuples = (1, 'a', 2, 'b')
type(tuples)
x = [2,3,'e', 'w', 9]
type(x)
type(1)
type(1.0)
x.append(3.4)
x
for item in x:

    print(item)
i=0

while(i != len(x)):

    print(i)

    i = i+1
[1,3] + [2,2]
[1]*4
2 in x
test = 'string manipulation using python'

print(test[0])
test[0:3]
test[-5]
test[-5:4]
test[-5:2]
test[-5:-2]
firstname = 'jitendra'

lastname = 'kasaudhan'

firstname*3
firstname[-1]
'jk' + 2
'jk' + str(2)
tpl = (1,3,'e',4,'a')
for item in tpl:

    print(item)
dict_eg = {'name': 'jk', 'age': '30'}
dict_eg['name']
for name, value in dict_eg.items():

    print(name)

    print(value)
for item in dict_eg.values():

    print(item)
# by default, it will loop through keys

for item in dict_eg:

    print(item)
tpl = ('jiten', '30', 'TUM')

name, age, uni = tpl
name
age
uni
sales_record = {'price': 2.5, 'num_items': 5, 'person': 'JK'}

statement = '{} bought {} products , with totoal price {}'

print(statement.format(sales_record['person'], sales_record['num_items'], sales_record['price'] * sales_record['num_items']))
# Reading CSV

import csv



with open('mpg.csv') as csv_file:

    data= list(csv.DictReader(csv_file))

    

# print first three items

print(data[:3])
len(data)
data[1]

data[0].keys() # print all keys of the dictionary

data[0].values() # print all values of the dictionary
# This is how to find the average cty fuel economy across all cars. All values in the dictionaries are strings, so we need to convert to float.

sum(float(d['cty']) for d in data) / len(data)
#Use set to return the unique values for the number of cylinders the cars in our dataset have.

cylinders = set(d['cyl'] for d in data)

cylinders
# datetime library

import datetime as dt

import time as tm



#time returns the current time in seconds since the Epoch. (January 1st, 1970)

tm.time()
dtnow = dt.datetime.fromtimestamp(tm.time())

dtnow
dtnow.year, dtnow.month, dtnow.hour, dtnow.minute, dtnow.second
#timedelta is a duration expressing the difference between two dates.

delta = dt.timedelta(days=100)

delta
diff = dt.date.today() - delta # 100 days ago

dt.date.today() > diff
#lamda functions are anonymous function i.e function without name for evaluating simple and short expressions

add_all = lambda a,b,c,d: a+b+c+d

add_all(2,2,2,2)

type(add_all)
# iterate from 0 to 100 and return even numbers

even_nums = []

for num in range(0, 100):

    if num % 2 == 0:

        even_nums.append(num)

        

print(even_nums)
# same as above but with list comprehension

even_nums = [num for num in range(0, 100) if num % 2 == 0 ]

even_nums
# list comprehension - find all possible combinations with two letter and two digits patter eg aa11

lowercase = 'abcdefghijklmnopqrstuvwxyz'

digits = '0123456789'



ans = [a+b+c+d for a in lowercase for b in lowercase for c in digits for d in digits]

ans[:10] # display first 10 items
len(ans)
import numpy as np

x = np.array([[1,2,3], [3,4,5]])

print(x)
y = np.array([[4,5,6], [7,8,9]])

print(x*y)
print(x+y)
# get dimension of the matrix

x.shape
ev = np.arange(0, 30, 2) # start at 0 count up by 2, stop before 30

ev
ev.reshape(3,5)
o = np.linspace(0,4,9)# return 9 evenly spaced values from 0 to 4

o
o.resize(3,3)

o
np.ones((3,2)) #ones returns a new array of given shape and type, filled with ones.
np.zeros((3,2)) #ones returns a new array of given shape and type, filled with zeros.
np.eye(3) #eye returns a 2-D array with ones on the diagonal and zeros elsewhere
np.diag(x) #diag extracts a diagonal or constructs a diagonal array
np.array([1,2,3] * 3) #Create an array using repeating list (or see np.tile)
np.repeat([1,2,3], 3) #Repeat each element of an array using repeat.
p = np.ones((2,3), int) # by default it creates floating ones but with int parameter, it created integer

p
np.vstack([p, p*2]) #Use vstack to stack arrays in sequence vertically (row wise).
np.hstack([p, p*2]) #Use hstack to stack arrays in sequence horizontally (column wise).
print(x**2) # elementwise power  [1 2 3] ^2 =  [1 4 9]
x = np.array([1,2,3])

y = np.array([[4],[5],[6]])

x.dot(y)
# Use .T to get the transpose.

x.T
x.T.shape
# Use .dtype to see the data type of the elements in the array.

x.dtype
# Use .astype to cast to a specific type

x.astype('f')

x.dtype
a = np.array([1,2,3,4,5,6])

a.sum()
a.max()
a.min()
a.mean()
a.std()
a.argmax() # argmax and argmin return the index of the maximum and minimum values in the array.
a.argmin() 
# Indexing / Slicing
s = np.arange(12)

s
s**2
#Quiz playground

['a', 'b', 'c'] + [1,2,3]
m1 = np.arange(36)

m1
m1.reshape(6,6)

m1
m1.reshape(36)[::7]
m1[::5] # get multiples of 5
m2 = np.arange(36)

changedShape = m2.reshape(6,6) # reshape does not change the original data structure but resize changes the data object

changedShape
changedShape[2] # returns 3rd row or 2nd index item
changedShape[1:] # returns all rows from index 1 and all columns
changedShape[[2,3,5]] # returns 2, 3 and 5 indexed row
changedShape[[2,3],[2,3]]