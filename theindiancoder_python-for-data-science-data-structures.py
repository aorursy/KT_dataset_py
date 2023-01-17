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
empty_list=[]

print(empty_list)



list_of_strings=['a','b','c']

print(list_of_strings)



list_of_numbers=[1,2,3,4,5]

print(list_of_numbers)



list_of_list=[list_of_strings, list_of_numbers]

print(list_of_list)



list_of_diff_data_type=['1',2,[3,4]]

print(list_of_diff_data_type)
lst=[1,2,3,4,5]

print(len(lst))



# len is a function
lst.append('appended item')

print(lst)
lst=[1,2,4]

lst.insert(2,'three')

# lst.insert(index, value)

# value at the specified index shifts

print(lst)
# if we want to use insert to add items at the end

lst.insert(4,'five')

print(lst)
lst=['one','two','three','four','five','two']

lst.remove('two') # it will always remove the first occurence of this element

print(lst)
# Append

# add the object at the end of the list

lst1=[1,2,3,4,5]

lst2=[6,7]

lst1.append(lst2)

print(lst1)
# Extend

# add elements of the second list at the end

# it will join the two lists

lst1=[1,2,3,4,5]

lst2=[6,7]

lst1.extend(lst2)

print(lst1)
# del 

lst=[1,2,3,4,5]

del lst[1]

print(lst)
# pop

lst=[1,2,3,4,5]

popped_item=lst.pop(1)

print('popped_item: ', popped_item)

print('list:', lst)

# remove: pass the element that you want to remove

lst=[1,2,3,4,5]

lst.remove(3)

print(lst)
x=['one','two','three']

if 'two' in x:

    print('two is there')

    

if 'four' not in x:

    print('four is not there')
lst=[1,2,3,4]

lst.reverse()

print(lst)
# In ascending order



lst=[3,1,6,2,8]

sorted_lst=sorted(lst)

print('list sorted in ascending order: ')

print(sorted_lst, end='\n\n')

print('original list: ')

print(lst)
# in descending order



des_sorted_lst=sorted(lst, reverse=True)

print('list sorted in descending order: ')

print(des_sorted_lst, end='\n\n')

print('original list: ')

print(lst)
lst=[3,1,6,2,8]

lst.sort()

print('Original list:', lst)

print('Sorted list',lst)
lst=[1,2,3,4,5]

# lst is pointing towards [1,2,3,4,5]'s memory location

abc=lst

# now, abc is also pointing towards [1,2,3,4,5]'s memory location



# we can call abc and lst as references to the same list



# appending 6 to abc

abc.append(6)

print('abc list:', abc)

print('lst list:', lst)





# Any edits we do with abc will be reflected in abc
tags='tag1 : tag2 : tag3 : tag4'

tag_list=tags.split(' : ')

print(tag_list)
# use case: split a sentence into words

s='This is a python tutorial'

word_list=s.split() # default is space

print(word_list)
lst=['one','two','three','four','five']

print('first element:',lst[0])

print('last element:',lst[-1])
#  0  1  2  3  # indexes from beginning 

# [1, 2, 3, 4] # numbers

# -4 -3 -2 -1  # indexes from the end
numbers=[10,20,30,40,50,60,70,80]



# all numbers

print('all the the numbers')

print(numbers[:])



# print numbers from index 0 to 3

print('print numbers from index 0 to 3')

print(numbers[0:4])
# alternate numbers in a list

print('alternate numbers in a list')

print(numbers[::2])

# from first end to the end with a step size of 2
# all numbers

print('all the the numbers')

print(numbers[:])

print(numbers[2::2])
lst1=[1,2,3]

lst2=['four','five','six']

print(lst1+lst2)



# combining both lists
numbers=[1,2,3,1,4,5,1,7,8]

# number of instances of 1

print('number of instances of 1:',numbers.count(1))



# number of instances of 3

print('number of instances of 3:',numbers.count(3))

lst=['one','two','three','four']

for item in lst:

    print(item)
# without using list comprehension

squares=[]

for num in range(10):

    squares.append(num**2)

print(squares)
# using list comprehension

squares=[i**2 for i in range(10)]

print(squares)
# using if with list comprehension

lst=[-10, -20, 30, 40, 50]

new_list=[i for i in lst if i>0]

print(new_list)
lst=[1,2,3,4,5]

new_list=[(i, i**2) for i in lst]

print(new_list)
# storing a matrix as a list of list

m=[  #c1    

    [ 1, 2, 3, 4], # row 1 of the matrix

    [ 5, 6, 7, 8],

    [ 9,10,11,12]

]



# transpose of a matrix



# mt=[

#     [1,5, 9],

#     [2,6,10],

#     [3,7,11],

#     [4,8,12]

# ]
# method 1



mt1=[]

for i in range(len(m[1])):

    sublist=[]

    for row in m:

        sublist.append(row[i])

    mt1.append(sublist)

print(mt1)
# method2 

[[row[i] for row in m] for i in range(4)]
# [row[1] for row in m]
# empty

t=()



# tuple of integers

t_int=(1,2,3)



# tuple of mixed data type

t_mixed=(1,'two','three')



# nested tuple

t_nested=(1, (1,2),[1,2,3])
t=('single')

print(t)

print(type(t))
t=('single',)

print(t)

print(type(t))
t='single',

print(t)

print(type(t))
t=('one','two','three','four')

print(t[1])
print(t[-1])
t=(1,(2,3,4))

print(t[1])
print(t[1][2])
t=(1,2,3,4,5,6)

print(t[1:4]) # from 2nd element to 4th element



print(t[:-2]) # upto 2nd last but not including



print(t[:]) # all elements
t=(1,2,3,[4,5,6])

t[3][1]='x'

print(t)
# t[3]='x'

# print(t)
(1,2,3)+(4,)
(1,2,3)*3
t=(1,2,3,40)

del t

# print(t)
t=(1,2,2,3,4,3,4,3)

print(t.count(3))
t=(1,2,2,3,4,3,4,3)

print(t.index(3))
t=(1,2,3,4,5)

print(1 in t)

print(7 in t)
#### Length

t=(1,2,3,4,(2,3))

len(t)
#### Sort

t=(7,4,9,3,6)

new_t=sorted(t)



# Returns a list



# does not alter the existing tuple

print(t)

print(new_t)
t=(7,4,9,3,6)

# min 

print('min:',min(t))



# max

print('max:',max(t))



# sum

print('sum:',sum(t))
s={1,2,3} # set elements

print(s)

print(type(s))
# every value is stored only once

s={1,2,3,1,4}

print(s)
# making a set from a list

s=set([1,2,3,1,4])

print(s)

print(type(s))
# initialize an empty set



s={}

print(type(s))



x=set()

print(type(x))
s={1,2,3}

# print(s[1])

# since they are unordered, they cannot be indexed
s={1,3}



# adding single elements



s.add(2)

print(s)
# adding multiple elements



# using update function



s.update([5,6,1]) # list

print(s)
s.update([7,8,9],{10,1,2})

print(s)
# discard

s={1,2,3,7,8}

s.discard(8)

print(s)
# remove

s.remove(7)

print(s)
# s.discard(7)

# s.remove(7)
# pop



# removes on element at random



s={1,2,3,4}

s.pop()

print(s)
# clear



# remove all the items from the set



s={1,2,3,4}

s.clear()

print(s)
s1={1,2,3,4}

s2={3,4,5,6}
print(s1 | s2)



print(s1.union(s2))
print(s1&s2)

print(s1.intersection(s2))
# elements which are there in s1 but not in s2

print(s1-s2)

print(s1.difference(s2))
print(s1^s2)

print(s1.symmetric_difference(s2))

print((s1|s2)-(s1&s2))
a={1,2,3,4}

b={3,4}

print('is a subset of b:', a.issubset(b))

print('is b subset of a:', b.issubset(a))
t1=(1,2)

t2=(1,2)

print(id(t1))

print(id(t2))
print(t1.__hash__())

print(t2.__hash__())
d={t1:'1'}

d[t2]
set1=frozenset([1,2,3,4])

set2=frozenset([3,4,5,6])

# set2.add(2)
# create a empty dict

d={}



# dict with integer keys

d={1:'abc',2:'xyz'}

print(d)



# dict with mixed keys

d={'name':'arun',10:[89,90,92]}

print(d)



# Internally, you can think of these as being stored in a table



# create an empty dict

d=dict()

print(d)



# create a dictionary with a list of tuples

d=dict([('name','abc'),(1,'xyz')])

print(d)
my_dict={'name':'Arun','age':27,'country':'India'}

print(my_dict['name']) # value corressponding to the key name

print(my_dict)
# print(my_dict['degree'])



# there is no key as degree

# gives key error
print(my_dict.get('name'))

print(my_dict.get('degree'))
# modifying values in dict



my_dict={'name':'Arun','age':27,'country':'India'}

my_dict['name']='raju'

print(my_dict)
# adding new values



my_dict['degree']='mba'

print(my_dict)

my_dict={'name':'Arun','age':27,'country':'India'}

print(my_dict)

removed_value=my_dict.pop('country')

print('removed value:',removed_value)



print(my_dict)
my_dict={'name':'Arun','age':27,'country':'India'}

print(my_dict)

my_dict.popitem()

print(my_dict)
my_dict={'name':'Arun','age':27,'country':'India'}

del my_dict['age']

print(my_dict)
my_dict.clear()

print(my_dict)
del my_dict

# print(my_dict)
# copy

my_dict={'name':'Arun','age':27,'country':'India'}

my_second_dict=my_dict.copy()

print(my_dict)

print(id(my_dict))

print(my_second_dict)

print(id(my_second_dict))



print('If we had not used copy method then the memory address would have been same', end='\n\n')



my_dict['degree']='MBA'

print(my_dict)

print(my_second_dict)



print('If we had not used copy, updating one would have also changed the other one')
# fromkeys



# create a dictionary with keys but same value



my_scores={}.fromkeys(['maths','english','science'],0)

print(my_scores)





my_scores={}.fromkeys(['maths','english','science'],[90,74,85])

print(my_scores)
my_dict={'name':'Arun','age':27,'country':'India'}

print(my_dict.items())



# returns a list of tuples of keys and values
print(my_dict.keys())
print(my_dict.values())
print(dir(my_dict))
print(my_dict)

d=my_dict.copy()

for pair in d.items():

    print(pair)
# example 1



new_d={k.upper():v for k,v in d.items()}

print(new_d)





# example 2



d={'num1':1,'num2':3,'num3':4,'num4':7}

print({k:v for k,v in d.items() if v>2})





# example 2



d={'num1':1,'num2':3,'num3':4,'num4':7}

print({k+'_':v*2 for k,v in d.items() if v>2})
print('Hello, world!')

print("Hello, Universe!")

print('''Hello, everyone!''')
s='Hello'

print(s[0])

print(s[-1])

print(s[3:])



# if index is out of range or a decimal number then we will get an error
my_str='Hello'

# my_str[0]='J'
# we can but delete the whole string

del my_str

# print(my_str)
s1='Hello'

s2='Arun'



# Concatenation

print(s1+' '+s2)



# Repeat the string n times

print('Ha'*4)
# Iterating through a string



# Count number of times the character appears

count=0

for letter in 'Hello there':

    if letter=='e':

        count+=1

print(count, 'e letters counts')
# Membership test

print('@' in 'myemail@emai.com')

print('@email.com' in 'myemail@email.com')
s='Hello'

print(s.lower())

print(s.upper())
s='This is a sentence'

words=s.split()

print(words)





# convert a string into a list

# each word as element of the list
# Reverse the above operation

new_s='*'.join(words)

print(new_s)
# Index where the first character matches

print('Good Morning'.find('@'))

print('Good Morning'.find('Morn'))
# Replace 

s='Bad Morning'

s1=s.replace('Bad','Good') # creating a new string. not replacing the original string

print(s)

print(s1)
my_str='Madam'

my_str=my_str.lower()

rev_str=reversed(my_str) # The reversed() method returns an iterator that accesses the given sequence in the reverse order.
rev_string=''.join(list(rev_str))

print(rev_string)
if rev_string==my_str:

    print('Palindrome')

else:

    print('not palindrome')
my_str='Program to sort words in alphabetical order'

word_list=my_str.split(' ')

words=[word.lower() for word in word_list]

print(words)
words.sort()

print(words)