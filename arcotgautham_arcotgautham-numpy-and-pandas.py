import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt
#1D numpy array

list1 = [1,2,3,4]

array = np.array(list)

print(array)



#2d numpy array

list2 = [2,3,4,5]

array2d = np.array([list1, list2])

print (array2d)



print ('the shape of the array is : \n', array.shape)

print ('the shape of the array2d is : \n', array2d.shape)



print ('the datatype of the array is :', array.dtype)

print ('the datatype of the array2d is : \n', array2d.dtype)



zeroArray = np.zeros(5)

print('1D zero array \n',zeroArray)



zeroArray2d = np.zeros([5,5])

print('2d zero array \n',zeroArray2d)



one_array = np.ones(5)

print ('1d unit array \n',one_array)



one_array2d = np.ones([5,5])

print('2d unit array \n',one_array2d)



emp_array = np.empty(5)

print('1d null array \n', emp_array)



emp_array2d = np.empty([5,5])

print('2d null array \n', emp_array2d)



#eye function

identy_array = np.eye(5)

print('identity matrix \n', identy_array)



#arange

#arthematic operation.

ap_array = np.arange(0,50,2)

print('arthematic progression numbers \n', ap_array)
#numpy scalar functions

#scalar op

#Scalar mul *

#expo Mul **

#sclar sub -

#sclar div /



import numpy as np





#scalar array mul

ar1 = np.array([[1,2,3],[3,4,5]])

print('ar1  ++++++++++++++++++++++')

print (ar1)

ar2 = np.array([[5,4,3],[3,2,1]])

print('ar2  ++++++++++++++++++++++')

print(ar2)



ar3 = ar1 * ar1

ar4 = ar2 * ar2

ar5 = ar1 * ar2

ar6 = ar3 * ar4

ar7 = ar5 * ar6

print('ar3  ++++++++++++++++++++++')

print(ar3)

print('ar4  ++++++++++++++++++++++')

print(ar4)

print('ar5  ++++++++++++++++++++++')

print(ar5)

print('ar6  ++++++++++++++++++++++')

print(ar6)

print('ar7  ++++++++++++++++++++++')

print(ar7)



#expo mul

print('square')

ar8 = ar1 ** 2

print(ar8)

print('cube')

ar9 = ar1 ** 3

print(ar9)



#scalar sub



ar10 = ar8 - ar9

print(ar10)



#scalar div

print('with number')

ar12 = 2/ar5

print(ar12)

print('two arrays')

ar11 = ar8/ar9

print(ar11)
#indexes in numpy



import numpy as np



arr = np.arange(100,160,2)

print (arr)



#intro to indexes 

#indexes are used to locate the value in a array

#example

#arr[0] means it will print the zeroth number in the given array

#i.e 100



print('element 1')

print(arr[0])

print('element 2')

print(arr[1])

print('element 3')

print(arr[2])

print('element 4')

print(arr[3])

print('element 5')

print(arr[4])



#slicing of array indexes arr[start:stop:step]

print(arr[0:5:2])



#updating array using slice

arr[0:5:3]=0

print (arr)



#slicing memory

arr2 = arr[4:10]

print(arr2)

arr2[:]=1

print(arr2)



print(arr)



#copy fucntion > new 



arr3 = arr.copy()

print(arr3)

arr3[:]=0

print(arr3)

print(arr)
import numpy as np



arr2d = np.array([[1,2,3],[2,3,4],[4,5,6]])

print(arr2d)



print(arr2d[0])

print(arr2d[1])



#accessing the particualr element

print(arr2d[1][1])



print(arr2d[1][2])



print (arr2d[2][2])



#slicing 2d array

slice1 = arr2d[0:1,0:2]

print(slice1)



slice2 = arr2d[0:2,0:3]

print(slice2)



slice3 = arr2d[:2,:3]

print(slice3)



slice4 = arr2d[1:,2:]

print(slice4)



#using loops to change index



arr_len = arr2d.shape

print(arr_len)

nrows = arr_len[0]

print (nrows)



for i in range(nrows):

    arr2d[i][i]=i

print(arr2d)





#accessing rows using list of index values



print(arr2d[[0,1]])



print(arr2d[[1,0]])
import numpy as np



#arange function

A = np.arange(15)

print(A)



B = np.arange(5,10)

print(B)



C = np.arange(5,10,2)

print(C)



#sqrt()



D = np.sqrt(A)

print(D)



#expo fun

E = np.exp(A)

print(E)



#random fun

F = np.random.randn(5)

print(F)



G = np.random.randn(5,5)

print(G)



#add function

print(A)

print(D)

H = np.add(A,D)

print(H)



#max function



N = np.array([1,3,5,6,9])

N1 = np.array([4,5,6,7,10])

H = np.maximum(N,N1)

print(H)
#saving and loading array



#why do we need to save array

#when we  are complex projects there are no. of problmes occur

#for instance we have 8gb ram in out pc, and sometimes if the pc

#if the program demands, more but pc cant give the memory there will be

#memory block



import numpy as np



arr = np.arange(10)

print(arr)



np.save('saved_array', arr)



load_arr1 = np.load('saved_array.npy')

print(load_arr1)



# saving multiple array



arr2=np.arange(25)

arr3=np.arange(5)



np.savez('saved_archive.npz', x=arr2, y=arr3)



#loading multiple array



load_npz = np.load('saved_archive.npz')

print(load_npz['x'])

print(load_npz['y'])



#saving array as text file



np.savetxt('myarray.txt', arr2, delimiter=',')

loadtxt=np.loadtxt('myarray.txt', delimiter=',')

print(loadtxt)



#delimiter: array is a sequence of values right so in order to spearate, the values

#we will use delimiter.



!cat myarray.txt
!pip3 install matplotlib



import numpy as np

import matplotlib.pyplot as plt



x = np.arange(3)

y = np.arange(4,8)

print(x)

print(y)



x2, y2 =np.meshgrid(x,y)



print (x2)



print(y2)



z = 2*x2 + 3*y2

print (z)



plt.imshow(z)

plt.title("plot of 2x+3y")

plt.colorbar()



#cos function

z2 = np.cos(x2)+np.cos(y2)

print(z2)

plt.imshow(z2)

plt.title("cos function for x2 and y2")

plt.colorbar()

plt.savefig('cosplt.png')



import numpy as np



x = np.array([100,400,-50,-40]) #each elemet a

y = np.array([10,14,23,30])# each elem b

condition = np.array([True, True, False, False])# each element cond



z = [a if cond else b for a,cond,b in zip(x, condition, y)]

print(z)



#Disadvantages

#using np.where(condition, values for true, values for false)



z2=np.where(condition,x,y)

print(z2)



z3 = np.where(x>0, 1,x)

print(z3)



#standard function

print(x.sum())



x2 = np.array([[1,2],[3,4]])

print(x2)

print('------------------------------')

print(x2.sum(0))



print(x.mean())

print(x.std())

print(x.var())



#logical AND, OR operations - any(), all()



cond2 = np.array([True, False, True])

#OR

print(cond2.any())

#AND

print(cond2.all())



#sort funcition



unsorted = np.array([1,5,6,12,63,10,24,673,346])

print (unsorted)

unsorted.sort()

print(unsorted)



#unique

ar2 = np.array(['liquid','solid','gas','liquid','solid','gas','liquid','solid','gas'])

print(np.unique(ar2))



#inid() list of elements are in aray or not



inid = np.in1d(['solid','gas','earth'], ar2)



print(inid)