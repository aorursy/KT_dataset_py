
#importing numpy package
import numpy as np

#checking numpy version
print('Installed NumPy version:',np.__version__)

#Creating list of integers
l=list(range(20))

#list in python works very much like arrays in C
print('\nComplete list\n','-'*30,'\n',l)

#Checking list data type
print('\ndata type of a list value:',type(l[0]))

#Accessing individual list elements
print('\n4th element of the list:',l[3])

#Creating list copy
l1=l.copy()

#A list can hold any data type as its element
#Adding items to an existing list
l.append(['a','b','c'])
l.append('xx')
l.append(0.3456)
print('\nComplete list\n','-'*30,'\n',l)
print('\nData type of 21st object in the list:',type(l[20]))
##Understanding Python list...

#unlike C array, python list can hold different data types within same list
l2=[1,'rahul',4.555,'amit','c']
print('Each element in List l2:\n',l2)

#creating a list of data types for each element in list l1
l2_dtype=[type(i) for i in l2]
print('\nData type of each element in List l2:\n',l2_dtype)
print('\nnotice both "amit" and "c" are considered string')
#Converting data type of each element of the list to string
l3=[str(i) for i in l]
print(l3)
##Operations on lists

#Add operator concatenates two lists
print('Adding two lists:\n',l+l1)

#Other mathematical operations will give error
#below line will give error

#print('Subtracting two lists:\n',l-l1) 
#importing array package
import array

#Creating an integer array
ar=array.array('i',l1)
ar
#output describes ar an array of type integer 'i'
#Creating array using numpy array() function
#range([start], stop[, step]) function start takes 0 as default
ar1=np.array(range(1,20,3))
ar1
##Python variables are pointers.

#When we assign one variable to another, we actually copy pointers or reference to other variable 
ar=ar1
print('array ar:\n',ar,type(ar))
for i in range(5):
    ar1[i]=ar1[i]+1
print('array ar1:\n',ar1)
print('array ar:\n',ar)

print('\nNote: Change done on array ar1 reflects in array ar')
##How to copy array

#Creating a copy of an array. And not just copying pointer value 
ar=np.array(ar1)
#can also use copy function ar=ar1.copy()

print('array ar:\n',ar)
for i in range(5):
    ar1[i]=ar1[i]+1
print('array ar1:\n',ar1)
print('array ar:\n',ar)

print('\nNote: In this case changes done on array ar1 not reflecting in array ar')
##Creating a blank array

#very useful when you don't know number of items you want to store
AR=np.array([])
AR1=np.array([])
AR2=np.empty((2,3),int)
print('blank array AR:\n',AR,',type:',type(AR))

#below assignment converts array type to list
AR=[1,2]
print('array AR:\n',AR,',type:',type(AR))
AR.append(range(5))
print('array AR:\n',AR,type(AR))

#Populating AR1 as an array
AR1=np.append(AR1,[0,1,2,3,4,5])
print('array AR1:\n',AR1,',type:',type(AR1))
##Creating a 2D array

AR2=np.array([range(i,i+5) for i in ar])
print(AR2)
#accessing individual element in 2D array
#remember index starts from AR2[0][0]
print('Value from 4th row and 4th Column:',AR2[3][3])
#creating a 2D array with variable number of row items
AR2_vr=np.array([range(i,i+100,i+3) for i in ar])
print(AR2_vr)
#we can also call this 1D array of variable length ranges
#accessing complete 2D array
for i in range(5):
    for j in AR2_vr[i]:
          print(j,end=',')
    print()
##Creating a 3D array

#1D array: collection of single items
#2D array: collection of 1D arrays
#3D array: collection of 2D arrays
AR3=np.array([AR2+i for i in range(3)])
print(AR3)
#accessing an element in 3D array
print('Value from 3rd row and 2nd Column of 2nd 2D array :',AR3[1][2][1])
##Creating a 4D array

#4D array: collection of 3D arrays
AR4=np.array([AR3+i for i in range(3)])
print(AR4)
#accessing an element in 4D array
print('Value from 3rd row and 2nd Column 2st 2D array within 2nd 3D array :',AR4[1][1][2][1])
##Creating a floating point array

#Random function generates floating value x (0.0 <= x < 1.0)
import random as rd
AR_fp=np.array([round(100*rd.random(),2) for i in range(10)])
print(AR_fp,'\n','Arrary data type:',type(AR_fp[0]))
#creating an array with combination of integer and floating point values
#all integers are upcasted as floating values
AR_fp1=np.array([1,2.3,3,4,5,6.7,8])
print(AR_fp1,'\n','Arrary data type:',type(AR_fp1[0]))
#declaring an array with initialization
AR_dz=np.zeros((3,5),dtype=int)
print(AR_dz)
#declaring an array with initialization
AR_do=np.ones((3,5),dtype=np.float64)
print(AR_do)
#declaring an array with given specific initialization
#We are initializing a float value, but downcasting using integer value
AR_ini=np.full((3,5),3.5,dtype=np.int32)
print(AR_ini)
#declaring an array of complex numbers
AR_i=np.full((3,5),3.5,dtype=np.complex64)
print(AR_i)
#arange() is numpy function which works same as range() function
AR_rng=np.arange(0,45,3)
print(AR_rng)
#using numpy linspace() function
#gives evenly spaced float values between start and stop
AR_lns=np.linspace(0,8,10)
print(AR_lns)
#using random() function with numpy
AR_nrd=np.random.random((3,3))
print(AR_nrd)
#creating a normal distribution (bell curve) with mean 0 and standard deviation 1
AR_nor=np.random.normal(0,1,(3,4))
print(AR_nor)
#creating a random array of integers
AR_rint=np.random.randint(0,20,(4,4))
print(AR_rint)
#creating identity matrix (array)
AR_imtx=np.eye(5)
print(AR_imtx)
#creating an empty array
#displayed values are values existing in memory
AR_emp=np.empty((3,4))
print(AR_emp)
#creating a diagonal matrix
AR_diag=np.diag((1,2,3,4))
print("Diagonal matrix with diagonal elements (1,2,3,4):\n",'-'*50,'\n',AR_diag)
##Creating Random Arrays

#seed function is used called while initializing Randomstate
#a constant value ensures same random combinations will be generated every time we run below code
#you can comment seed() function to generate different combinations each time
np.random.seed(1)
x1=np.random.randint(-15,15,size=10)
print('\n1D random array\n','-'*25,'\n',x1)
x2=np.random.randint(100,size=(3,4))
print('\n2D random array\n','-'*25,'\n',x2)
x3=np.random.randint(10,size=(3,4,6))
print('\n3D random arrary\n','-'*25,'\n',x3)
##Array Attributes

#ndim gives array dimensionality
print('dimensionality of x1, x2, x3:',x1.ndim,'D,',x2.ndim,'D,',x3.ndim,'D')
#size gives number of elements in an array
print('size of x1, x2, x3:',x1.size,',',x2.size,',',x3.size)
#shape gives number of rows, columns, etc. in a given array
print('shape of x1, x2, x3:',x1.shape,',',x2.shape,',',x3.shape)
#dtype gives data type of an array
print('data type of x1, x2, x3:',x1.dtype,',',x2.dtype,',',x3.dtype)
#itemsize gives byte size of each array element
print('Each item size in bytes of x1, x2, x3:',x1.itemsize,' bytes,',x2.itemsize,' bytes,',x3.itemsize,' bytes')
#nbytes gives byte size of an array
print('Array size in bytes of x1, x2, x3:',x1.nbytes,' bytes,',x2.nbytes,' bytes,',x3.nbytes,' bytes')
#nbyte calculation (nbyte=size*itemsize)
bsize_x1=x1.size*x1.itemsize
bsize_x2=x2.size*x2.itemsize
bsize_x3=x3.size*x3.itemsize
print('Array size in bytes of x1, x2, x3:(size*itemsize)',bsize_x1,' bytes,',bsize_x2,' bytes,',bsize_x3,' bytes')
##Array Flags

#C_CONTIGUOUS: The data is in a single, C-style contiguous segment
#F_CONTIGUOUS: The data is in a single, Fortran-style contiguous segment
#OWNDATA: The array owns the memory it uses or borrows it from another object
#WRITEABLE: The data area can be written to. Setting this to False locks the data, making it read-only
#ALIGNED: The data and all elements are aligned appropriately for the hardware
#UPDATEIFCOPY: This array is a copy of some other array. 
#When this array is deallocated, the base array will be updated with the contents of this array
print('Flags associated with array x1:\n','-'*50,'\n',x1.flags)
#setting WRITEABLE FLAG to false
#Array items can't be edited if WRITEABLE flag is false
x1.flags.writeable=0
print('x1 flags after setting WRITEABLE flag to false:\n','-'*50,'\n',x1.flags)
##Understanding array indices

print('Array x1:',x1)
print('Arrary x1 3rd element:',x1[2])

#Negative array indices
#negative indices are used to read array in reverse order. x1[-1] will fetch last element
print('last element of x1:',x1[-2])

#indices for 2D array
print('Array x2:\n',x2)
print('Array x2 element at 2nd row and 3rd column:',x2[1,2])

#indices for 3D array
print('Array x3:\n',x3)
print('Array x3 element at 2nd 2D array 2nd row and 3rd column:',x3[1,1,2])
##Slicing Arrays
#Slicing 1D Array

#Array slicing uses following syntex x[start:stop:step]
print('Array x1:',x1)

#Slicing a 1D array (fetching alternate elements)
s_x1=x1[0:x1.size:2]
print('Sliced x1 array:',s_x1)

#accessing reversed array using slicing
rev_x1=x1[::-1]
print('Reverse x1:',rev_x1)

#using negative value as step
revi_x1=x1[6::-2]
print('Reverse step x1:',revi_x1)
#Slicing 2D Array

#2D array x2
print('Array x2:\n',x2)

#accessing complele 2D array using slicing
print('Accessing complete array x2 using slicing:\n',x2[:,:])

#Excluding first row of 2D array x2
print('Excluding first row of array x2:\n',x2[1:,:])

#accessing alternte rows of 2D array
print('Accessing alternate rows of array x2:\n',x2[::2,::])

#accessing alternte columns of 2D array
print('Accessing alternate cols of array x2:\n',x2[::,::2])

#accessing last row of  2D array using negative indexing
print('Accessing last row of array x2:\n',x2[-1::,::])

#accessing last 2 cols of 2D array using negative indexing
print('Accessing last 2 cols of array x2:\n',x2[::,-2::])
##Combining indices and slicing

print('Array x2:\n',x2)

#accessing 3rd col of array x2 by combining slicing and indices
print('Accessing 3rd col of Array x2:\n',x2[:,2])

#accessing last 2 elements of 4th col from array x2 by combining slicing and indices
print('Accessing last 2 elements of 4th col from Array x2:\n',x2[1:,3])
##Reshaping Arrays

#Reshaping 1D array into 2D array
print('Array x1:\n',x1)
print('Reshaped Array x1:\n',x1.reshape((2,5)))

#Reshaping 2D array into 1D array
print('Array x2:\n',x2)
print('Reshaped Array x2 - 2D to 1D:\n',x2.reshape((x2.shape[0]*x2.shape[1])))

#Reshaping 2D array into 2D array of different shape 
print('Reshaped Array x2 - 2D changed shape:\n',x2.reshape((x2.shape[1],x2.shape[0])))


print('Note: Number elements in reshaped array must be same as base array')
#below line will through error
#print('Reshaped Array x1:',x1.reshape((4,2)))
##Array Concatenation

print('Array x1:\n',x1)
print('Concatenating Array x1 with Array x1:\n',np.concatenate([x1,x1]))
print('Note: Only Arrays with same number of dimensions can be concatenated')
#therefore below code will through error
#print('Concatenating Array x1 with Array x2:\n',np.concatenate([x1,x2]))

#concatenating two 2D arrays
print('Array x2:\n',x2)
print('Concatenating x2 with x2:\n',np.concatenate([x2,x2]))

#horizontally concatenating two 2D arrays
print('Horizontally concatenating x2 with x2:\n',np.hstack([x2,x2]))
##Array Splitting

#splitting 1D array
print('Array x1:\n',x1)
print('Splitting Array x1 at index 4 and 7:\n',np.split(x1,[4,7]))

#vertically splitting 2D array
print('Array x2:\n',x2)
print('Splitting x2 vertically into individual rows:\n',np.vsplit(x2,[1,2]))

#horizontally splitting 2D array
print('Splitting x2 horizontally into 3 parts:\n',np.hsplit(x2,[1,2]))
print('Splitting x2 horizontally into 4 parts:\n',np.hsplit(x2,[1,2,3]))
print('Note: If we give N indices to split function N+1 arrays will be generated')