import numpy as np

#list 

list1=[1,2,3,4,5]
list1
list2=[6,7,8,9,10]
#using numpy creating mutidimensional array 

add= np.array([list1,list2])
add
add.shape
type(add)

#Re-shaping the structure of 2d array  into 2 columns and 5 arrays .

reshape_now=add.reshape(5,2)
reshape_now
#accessing the fifth Eelement

add[1]

reshape_now[4]
reshape_now[:,:]
reshape_now[:]
# creating a new list using numpy

new_list1=np.arange(0,10)
new_list1
# creating a new list using numpy

new_list2=np.arange(0,10, step=3)
new_list2
#it will repaete all elemsts from the selection element to onwords .

new_list2[2:]=40
new_list2
# generating random values in specific range

np.linspace(1,5,30)
np.arange(0,10).reshape(2,5)
np.ones((2,5),dtype=int)
#generating random 2 rows of 5 columns

np.random.randn(2,5)