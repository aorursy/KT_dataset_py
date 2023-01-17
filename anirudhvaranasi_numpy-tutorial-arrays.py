import numpy as np
# Creating Numpy Array

my_list = [1, 2, 3]
x = np.array(my_list)
type(x)
my_matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
np.array(my_matrix)
list(range(0, 5))
np.arange(0, 5)
np.arange(0, 11, 2)
np.zeros(3)
type(1)
type(1.0)
np.zeros(4)
np.zeros((5,5))
np.ones(4)
np.ones((3,3))
np.linspace(0, 10, 51)
np.eye(4)
np.random.rand(5,4)
np.random.randn(5, 4)
np.random.randint(1, 100, 10)
arr = np.arange(25)

ranarr = np.random.randint(0, 50, 10)
arr

ranarr
arr.reshape(5, 5)
arr.reshape(25, 1).shape
arr.dtype
ranarr
ranarr.max()
ranarr.min()
ranarr.argmax()
ranarr.argmin()