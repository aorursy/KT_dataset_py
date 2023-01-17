# Hello world

print('Hello_world')
#try yourself

print('---')
#Assignments

#As in other languages no need to explicitly define the data type of the variable :D



#strings

print('Strings')

DATA ='HELLO WORLD'

print(DATA[0])

print(len (DATA))

print(DATA)

print('-'*30)



# NUMBERS

print('Numbers')

VALUE = 123.1

print(VALUE)

VALUE = 10

print(VALUE)

print('-'*30)



# BOOLEAN

print('Boolean')

A = True

B = False

print(A, B)

#Flow Control

#If-else statements

print('If-else statements')

VALUE = 99

if VALUE == 99:

    print('THAT IS FAST')

elif  VALUE > 200:

    print('THAT IS TOO FAST')

else:

    print('THAT IS SAFE')
#FOR LOOP

print('-'*30)

print('FOR LOOP')

for I in range(10):  

    print(I)

    

#WHILE LOOP

print('-'*30)

print('WHILE LOOP')

I = 0

while I < 10:  

    print(I)

    I += 1
#Data Structure

#tuples

print('Tuples')



a = (1,2,3,4,'a','b')

print(a)

b = a

print('length of b is : ',len(b))
#Data Stucture

#List

list1 = [1,2,3,'hello',True,'world']

list2 = list1

print(list1)

print('length of list2 is :' ,len(list2))

print('Third element of list1 : ',list1[2])

print('see the mutation')

list2[2] = 'baymax'

print('Third element of list1 after mutation : ',list1[2])

#Data tructure

#Dictionaries



#This is dictionary of names and age of a group of people

dict1 = {'Ram':25,'Gokul':22,'Joseph': 55,'Ronald':43,'Bilal':23,'Dev':21}

print("Ram's age is: ", dict1['Ram'])

dict1['Ram'] = 20

print("Ram's correct age is:",dict1['Ram'])



for key in dict1.keys():

    print(key," ",dict1[key])
#Functions

#Function add sum of two number 



def sum(a,b):

    return a+b



print(sum(2,3))
#Operators
#File I/O
#Numpy

# define an array

import numpy as np

mylist = [1, 2, 3,4,6,7,99,3]

myarray = np.array(mylist)

print(myarray)

print(myarray.shape)



#Numpy useful functions

print('sum :',np.sum(myarray))

print('mean :',np.mean(myarray))

print('Standard Deviation :',np.std(myarray))



# access values

import numpy

mylist = [[1, 2, 3], [3, 4, 5]]

myarray = numpy.array(mylist)

print(myarray)

print(myarray.shape)

print("First row: ",myarray[0])

print("Last row: ",myarray[-1])

print("Specific row and col: ",myarray[0, 2])

print("Whole col: ",myarray[:, 2])
# arithmetic

import numpy

myarray1 = numpy.array([2, 2, 2])

myarray2 = numpy.array([3, 3, 3])

print("Addition: ",(myarray1 + myarray2))

print("Multiplication:",(myarray1 * myarray2))
# basic line plot

import matplotlib.pyplot as plt

import numpy

myarray = numpy.array([1, 2, 3])

plt.plot(myarray)

plt.xlabel('some x axis')

plt.ylabel('some y axis')

plt.show()
# basic scatter plot

import matplotlib.pyplot as plt

import numpy

x = numpy.array([1, 2, 3,4,5,6])

y = numpy.array([2, 4, 6,8,10,12])

plt.scatter(x,y)

plt.xlabel('some x axis')

plt.ylabel('some y axis')

plt.show()
# series

import numpy

import pandas

myarray = numpy.array([1, 2, 3])

rownames = ['a','b','c']

myseries = pandas.Series(myarray, index=rownames)

print(myseries)



print(myseries[0])

print(myseries['a'])
# dataframe

import numpy

import pandas

myarray = numpy.array([[1, 2, 3], [4, 5, 6]])

rownames = ['a','b']

colnames = ['one','two','three']

mydataframe = pandas.DataFrame(myarray, index=rownames, columns=colnames)

print(mydataframe)