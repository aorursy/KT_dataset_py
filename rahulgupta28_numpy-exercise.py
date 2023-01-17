#NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices.

#Along with a large collection of high-level mathematical functions to operate on these arrays.
import numpy as np 

print(np.__version__)

print(np.show_config)
my_list = [1,2,3,4]

np.array(my_list)
list_of_list = [['a','b','c'],[1,2,3]]

np.array(list_of_list)


my_dict = {'a':1,'b':2,'c':3,'d':4}

np.array(my_dict)
data = [[1,2,3,4,5],('a','b','c','d'),{'a':1,'b':2,'c':3,'d':4},[7,8,9,10]]

np.array(data)
matrix = [[1,2,3],[4,5,6],[7,8,9]]

matrix
np.array(matrix)
np.arange(7)
#generate array from '0'(include) to '10' (exclude)

np.arange(0,10)
#generate array from '0'(include) to '10' (exclude) with step size '2' and datatype =float

np.arange(start= 0,stop = 10,step= 2, dtype=float)
np.add(5,7)
L1 = np.arange(1,10,2)

L2 = np.arange(0,10)

np.add(L1,L2)
L1 = np.arange(1,20,2) # odd number

print(L1)

L2 = np.arange(0,10)

print(L2)

np.add(L1,L2)
matrix_1 =np.arange(9).reshape(3,3)

print(matrix_1)

print('---------------------')

matrix_2 = np.arange(3)

print(matrix_2)

print("--------------------")

np.add(matrix_1,matrix_2)

print(matrix_1.shape)

print(matrix_2.shape)
np.linspace(start = 0,stop = 3)  #bydefault no. of Sample size = 50 
np.linspace(start = 2 , stop = 3 , num =5)
np.linspace(7,9,5,retstep=True)
np.zeros(shape=5)
np.zeros(shape =8 ,dtype = int)
np.zeros(shape=(2,2))
np.ones(7)
np.ones(7,dtype=int)
np.ones(shape=(2,2))
np.eye(6,dtype=int)
np.eye(6,k=1)
np.eye(6,k=-1)
#To Create an array of the given shape & populate , it with random sample from a "uniform distribution" over (0,1)

np.random.rand(2)
np.random.rand(4,4)
#To Create an array of the given shape & populate , it with random sample from a  "Standard Normal "distribution.

print(np.random.randn())

print('-------------------------------------------')

print(np.random.randn(5))
np.random.randn(3,3)
#To Create an array of the given shape & populate , it with random sample from a  "Discrete uniform " distribution.

#retun random integer from low(inclusive) to high (exclusive)



print(np.random.randint(10))

print("---------------------------------------")

print(np.random.randint(low = 2,high = 20 ,size =10 ))

print('------------------------------------------------')
print(np.random.randint(low = 2,high = 20 ,size =10 ,dtype = float))
np.random.randint(5,size = (3,3))