import numpy as np
# List of functions explained 

function1 = np.array

function2 = np.concatenate

function3 = np.ones

function4 = np.matmul

function5 = np.eye



"""

Add your function name here

"""
# Example 1 - working 

np.array([[5,10,15,20],[3,6,9,12]])
# Example 2 - working 

np.array([[1,2,3,4,5],[6,7,8,9,10.0]])
np.array([[1,2,3],[4,5]],dtype=int)
# Example 1 - working 

arr1 = [[1, 2], 

        [3, 4.]]



arr2 = [[5, 6, 7], 

        [8, 9, 10]]



np.concatenate((arr1, arr2), axis=1)
# Example 2 - working

a = np.array([[1, 2], [3, 4]])

b = np.array([[5, 6]])



np.concatenate((a, b), axis=None)
# Example 3 - breaking (to illustrate when it breaks)

arr1 = [[1, 2], 

        [3, 4.]]



arr2 = [[5, 6, 7], 

        [8, 9, 10]]



np.concatenate((arr1, arr2), axis=0)
# Example 1 - working

np.ones(5,dtype=int)
# Example 2 - working

np.ones((2,3))
# Example 3 - breaking (to illustrate when it breaks)

np.ones(3,4)
a = np.array([[1, 0],

              [0, 1]])

b = np.array([[4, 1],

              [2, 2]])

np.matmul(a, b)

a = np.array([[1, 0],

              [0, 1]])

b = np.array([1, 2])

np.matmul(a, b)



np.matmul(b, a)
# Example 3 - breaking (to illustrate when it breaks)

np.matmul([1,2], 3)
np.eye(5)
np.eye(3,5,dtype=int)
np.eye(2,3.5,dtype=float)
#Example 1



#Example 2



#Example 3 aka the "code-phat-gya" example



"""

Add your code cells below this cell...

DO NOT DELETE ABOVE CODE CELLS...

"""