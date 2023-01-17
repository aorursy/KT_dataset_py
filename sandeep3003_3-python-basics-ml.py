x = 1

y = 543216789123456789

z = -9876543210



print(type(x))

print(type(y))

print(type(z))
a = 1.2

b = 321654.0987

c = -0.678954321

print(type(a))

print(type(b))

print(type(c))
p = "Hello, World"

q = """Lorem ipsum dolor sit amet,

consectetur adipiscing elit,

sed do eiusmod tempor incididunt

ut labore et dolore magna aliqua"""

r = "wrap lot's of other quotes"

print(p)

print('\n') #added a new line

print(q)

print('\n') #added a new line

print(r)
print('Hello, World!')

print('\n') #added an new line

w = "Hi, We're trying print some statements"

print(w)

print('\n')

name = 'Python'

age = 30

print('My name is: {one}, and my age is: {two}.'.format(one=name,two=age))

print('\n')

print('My name is: {}, and my age is: {}.'.format(name, age))

print('\n')

print(f"My name is: {name}, and my age is: {age}.")
# assigning list to a variable

thislist = ["apple", "banana", "cherry", "orange", "kiwi", "melon", "mango"]

# print the List

print(thislist)

# thislist is a list
# printing a string using the index of a list

print(thislist[1])
# printing a string from reverse the order or statring from left to right indexing

print(thislist[-1])
# Slicing

#    -7        -6        -5        -4       -3      -2       -1

# ['apple', 'banana', 'cherry', 'orange', 'kiwi', 'melon', 'mango']

#     0         1         2         3        4       5        6





# this will take the string from index starting from 2 upto and excluding index 5

print(thislist[2:5])
#    -7        -6        -5        -4       -3      -2       -1

# ['apple', 'banana', 'cherry', 'orange', 'kiwi', 'melon', 'mango']

#     0         1         2         3        4       5        6



#this will take the string from index starting from 0 upto and excluding index 4

print(thislist[:4])
#    -7        -6        -5        -4       -3      -2       -1

# ['apple', 'banana', 'cherry', 'orange', 'kiwi', 'melon', 'mango']

#     0         1         2         3        4       5        6



#this will take the string from index starting from 2 upto to the last index but it will exclude those index for 0 to till index 2

print(thislist[2:])
#    -7        -6        -5        -4       -3      -2       -1

# ['apple', 'banana', 'cherry', 'orange', 'kiwi', 'melon', 'mango']

#     0         1         2         3        4       5        6



# #this will take the string from index starting from -4 upto and excluding index -1 in backward counting of the index

print(thislist[-4:-1])
# assigning or re-assigning a string to the index

thislist[1] = "blackcurrant"

print(thislist)
# iterrating a list in a for loop

for x in thislist:

    print(x)
# checking if the string is present on the list

if "apple" in thislist:

    print("Yes, 'apple' is in the fruits list")
print(len(thislist))
# append is used to insert a variable at the end of the list

thislist.append("orange")

print(thislist)
# insert is used to insert a variable at any index of the length of the list

thislist.insert(1, "orange")

print(thislist)
# this will remove a specific variable from the list

thislist.remove("orange")

print(thislist)
# this is remove the very last index from the list

thislist.pop()

print(thislist)
# delete a variable from a specific index of a list

del thislist[0]

print(thislist)
# it deletes the complete list

del thislist
thislist = ["apple", "banana", "cherry", "orange", "kiwi", "melon", "mango"]
# list will exist but all the index's variable will be deleted

thislist.clear()

print(thislist)
thislist = ["apple", "banana", "cherry", "orange", "kiwi", "melon", "mango"]

# this copies the list to another variable

mylist = thislist.copy()

print(mylist)
mylist = list(thislist)

print(list(thislist))
# by doing this, whenever there is a change in thislist the same change will be shown in mylist too

mylist = thislist
thislist.insert(3, 'grapes')

print (thislist)
print(mylist)
# A list can contain any kind of datatype within it

thislist.append(100000)

print (thislist)
# even a list inside a list/nested list

thislist.append(['dragon fruit', '2020'])

print (thislist)
# creating and printing dictionary

thisdict = {

  "brand": "Ford",

  "model": "Mustang",

  "year": 1964

}

print(thisdict)
# accessing dictionary

thisdict["model"]

thisdict.get("model")
# Changing the value for a ke

thisdict["year"] = 2018

thisdict["year"]
# looping a dictionary (keys)

print ('looping keys:')

for x in thisdict:

    print(x)



# looping a dictionary (values)

print ('looping values:')

for x in thisdict:

    print(thisdict[x])
# looping values of a dictionary

for x in thisdict.values():

    print(x)
# looping key, value pair

for x, y in thisdict.items():

    print(x, ":", y)
# check if key exist in the dictionary

if "model" in thisdict:

    print("Yes, 'model' is one of the keys in the thisdict dictionary")
# print the length dictionary

print(len(thisdict))
# adding a new key and value pair

thisdict["color"] = "red"

print(thisdict)
# removing a item from dictionary

thisdict.pop("model")

print(thisdict)
# nested dictionary

child1 = {

  "name" : "Emil",

  "year" : 2004

}

child2 = {

  "name" : "Tobias",

  "year" : 2007

}

child3 = {

  "name" : "Linus",

  "year" : 2011

}



myfamily = {

  "child1" : child1,

  "child2" : child2,

  "child3" : child3

}

print(myfamily)
# dictionary as constructor

thisdict = dict(brand="Ford", model="Mustang", year=1964)

# note that keywords are not string literals

# note the use of equals rather than colon for the assignment

print(thisdict)
print(10 > 9)

print(10 == 9)

print(10 < 9)
a = 200

b = 33



if b > a:

    print("b is greater than a")

else:

    print("b is not greater than a")
thistuple = ("apple", "banana", "cherry")

print(thistuple)
# accesing tuple items

thistuple = ("apple", "banana", "cherry")

print(thistuple[1])
# range of indexes

thistuple = ("apple", "banana", "cherry", "orange", "kiwi", "melon", "mango")

print(thistuple[2:5])
# cant change the value once assigned to a list

x = ("apple", "banana", "cherry")

x[1] = "kiwi"
thisset = {"apple", "banana", "cherry"}

print(thisset)
thisset = {"apple", "banana", "cherry"}



for x in thisset:

    print(x)
# join to sets

set1 = {"a", "b" , "c"}

set2 = {1, 2, 3}



set3 = set1.union(set2)

print(set3)
# updating two sets

set1 = {"a", "b" , "c"}

set2 = {1, 2, 3}



set1.update(set2)

print(set1)
# set as constructor

thisset = set(("apple", "banana", "cherry")) # note the double round-brackets

print(thisset)
a = 33

b = 200

if b > a:

    print("b is greater than a")
a = 33

b = 33

if b > a:

    print("b is greater than a")

elif a == b:

    print("a and b are equal")
a = 200

b = 33

if b > a:

    print("b is greater than a")

elif a == b:

    print("a and b are equal")

else:

    print("a is greater than b")
# single line if Statement

if a > b: print("a is greater than b")
# single line else if

a = 2

b = 330

print("A") if a > b else print("B")
#single line else if with condition

a = 330

b = 330

print("A") if a > b else print("=") if a == b else print("B")
a = 33

b = 200



if b > a:

    pass
fruits = ["apple", "banana", "cherry"]

for x in fruits:

    print(x)
# looping a string

for x in "banana":

    print(x)
fruits = ["apple", "banana", "cherry"]

for x in fruits:

    print(x)

    if x == "banana":

        break
fruits = ["apple", "banana", "cherry"]

for x in fruits:

    if x == "banana":

        break

    print(x)
fruits = ["apple", "banana", "cherry"]

for x in fruits:

    if x == "banana":

        continue

    print(x)
for x in range(6):

    print(x)
for x in range(2, 6):

    print(x)
for x in range(2, 30, 3):

    print(x)
for x in range(6):

    print(x)

else:

    print("Finally finished!")
adj = ["red", "big", "tasty"]

fruits = ["apple", "banana", "cherry"]



for x in adj:

    for y in fruits:

        print(x, y)
i = 1

while i < 6:

    print(i)

    i += 1
i = 1

while i < 6:

    print(i)

    if i == 3:

        break

    i += 1
i = 0

while i < 6:

    i += 1

    if i == 3:

        continue

    print(i)
i = 1

while i < 6:

    print(i)

    i += 1

else:

    print("i is no longer less than 6")
# creating a loop inside a list avoiding multiple lines of looping and append function

x = [1,2,3,4]

[item**2 for item in x]
def my_func(param1='default'):

    """

    Docstring goes here.

    """

    print(param1)
my_func
# Default Parameter Value

my_func()
# Arguments

my_func('new param')
# Keyword Arguments

my_func(param1='new param')
def my_function(*kids):

    print("The youngest child is " + kids[2])



my_function("Emil", "Tobias", "Linus")
def my_function(**kid):

    print("His last name is " + kid["lname"])



my_function(fname = "Tobias", lname = "Refsnes")
# A lambda function that adds 10 to the number passed in as an argument, and print the result:

x = lambda a : a + 10

print(x(5))
# A lambda function that multiplies argument a with argument b and print the result:

x = lambda a, b : a * b

print(x(5, 6))
# A lambda function that sums argument a, b, and c and print the result:

x = lambda a, b, c : a + b + c

print(x(5, 6, 2))
def myfunc(n):

    return lambda a : a * n

mydoubler = myfunc(2)

print(mydoubler(11))
my_list = [1, 5, 4, 6, 8, 11, 3, 12]



new_list = list(map(lambda x: x * 2 , my_list))



print(new_list)
my_list = [1, 5, 4, 6, 8, 11, 3, 12]



new_list = list(filter(lambda x: (x%2 == 0) , my_list))



print(new_list)