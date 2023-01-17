import numpy as np
my_list = [1,2,3]

my_list
np.array(my_list)
my_matrix = [[1,2,3],[4,5,6],[7,8,9]]

my_matrix
np.array(my_matrix)
np.arange(0,10)
np.arange(0,11,2)
np.zeros(3)
np.zeros((5,5))
np.ones((3,3))
np.linspace(0,10,10)
np.eye(4)
np.random.rand(2)
np.random.rand(5,5)
np.random.randn(2)
np.random.randn(5,5)
np.random.randint(1,100)
np.random.randint(1,100,10)
arr = np.arange(25)

ranarr = np.random.randint(0,50,10)
arr
ranarr
arr.reshape(5,5)
# Vector

arr.shape
# Notice the two sets of brackets

arr.reshape(1,25)
arr.reshape(1,25).shape
arr.reshape(25,1)
arr.reshape(25,1).shape
arr.dtype