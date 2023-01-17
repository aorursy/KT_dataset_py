import numpy as np



x = np.array([1, 3, 5, 7])



type(x)



np.array([(1,3,5), (7, 9, 11), (13, 15, 17)])

np.array([(1,3,5), (7, 9, 11), (13, 15, 17)]).ndim



np.zeros((2, 3), dtype=np.int16)

np.ones((2, 3), dtype=np.int16)



np.empty((2, 2, 4))



np.linspace(2, 4, 10)



np.random.random((3, 3))



threeD = np.arange(1, 30, 2).reshape(3, 5)

threeD



threeD.itemsize



a = np.arange(5)

b = np.array([2, 4, 0, 1, 2])

np.sin(a)

np.sum(a)

np.max(a)

b > 2



x = np.array([[1, 1], [0, 1]])

y = np.array([[2, 0], [3, 4]])

x * y

np.dot(x, y)



np.mean(x)
data_set = np.random.random((5, 10))

data_set
data_set[2:4, 0:2]
data_set[:, 0]
data_set[2:5:2]