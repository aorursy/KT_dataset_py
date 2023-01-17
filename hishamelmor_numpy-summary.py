import numpy as np
np.zeros((3,6))
np.ones((3,6))
np.arange(0,12,1.6)
np.linspace(0,3,100)
np.eye(5)

#Identity matrix
np.random.rand(3)
np.random.randn(2,2)

# norman distribution around 1
np.random.randint(1,100)
np.random.randint(1,100,3)
arr = np.random.randint(0,20,25)
arr.reshape(5,5)
arr.max()
arr.argmax()

#the index of maximum value
arr.shape
arr = arr.reshape(5,5)
arr.shape
arr[:3]
copy=arr.copy()
copy.reshape(25)
copy
arr_2d = np.array([[1,2,3],[4,5,6],[7,8,9]])
arr_2d
arr_2d[1][1]

#indexing two dimentional array
arr_2d[0]

#complete row indexing
arr_2d[:2,1:]

#slicing 
array = np.arange(1,11)
array
array >=5
array[array<=3]
array - array
array