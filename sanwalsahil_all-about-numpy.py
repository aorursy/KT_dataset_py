# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra





# Any results you write to the current directory are saved as output.
data = [1,2,3,4,5]

arr = np.array(data)

arr
data = [[1,2,3,4],[5,6,3,2]]

arr = np.array(data)

print(arr.shape)

arr
arr.ndim
arr.shape
arr.dtype
arr = np.zeros(10)

print(arr.shape) 

print(arr.dtype)

print(arr.ndim)

arr
arr = np.zeros((2,3))

print(arr.shape)

print(arr.dtype)

print(arr.ndim)

arr
arr = np.zeros((2,3,2))

print(arr.shape)

print(arr.dtype)

print(arr.ndim)

arr
arr = np.ones((2,3,2))

print(arr.shape)

print(arr.dtype)

print(arr.ndim)

arr
np.arange(5)
arr= np.array([1,2,3],dtype=np.float64)

arr
arr = np.array([1,2,3])

print(arr.dtype)

new_arr = arr.astype(np.float64)

print(new_arr.dtype)
arr = np.array([[1,2],[3,4]])

print(arr*arr)

print('________________________')

print(arr)
arr1 = np.array([1,2])

arr2 = np.array([2,4])

print(arr1*arr2)
arr = np.array([1,2])

print(arr*2)
arr1 = np.array([1,2])

arr2 = np.array([2])

print(arr1*arr2)

print(arr1.shape)

print(arr2.shape)
arr1 = np.array([1,2])

arr2 = np.array([2,3,5])



print(arr1.shape)

print(arr2.shape)

print(arr1*arr2)
arr = np.array([1,2,3])

arr1 = np.array([2,3,4])

print(arr1-arr)
arr = np.array([1,2])

arr1 = np.array([2,3,4])

print(arr1-arr)
arr = np.array([1])

arr1 = np.array([2,3,4])

print(arr1-arr)
arr = np.array([2,4,6])

newArr = arr/2

print(newArr)
arr1 = np.array([1,2,3])

arr2 = np.array([1,1,1])

print(arr1>arr2)
arr = np.arange(10)

arr[5:10]
arr[2:]
arr[:2]
arr[:-2]
arr[-2:]
arr[2:-3]
arr[-2:5]
arr = np.array([[1,2,3],[2,3,4],[5,4,3],[1,1,1]])

print(arr.shape)

print(arr.ndim)

arr
arr[0]
arr = np.array([1,2,3,4,5,6])

arr[arr > 2]
arr[arr > 2 | arr < 4]
arr = np.arange(15).reshape((3,5))

arr
arr.T
arr = np.arange(10)

np.sqrt(arr)
np.exp(arr)
np.random.rand(4)
arr1 = np.array([1,2,3,4])

arr2 = np.array([1,3,2,4])

np.maximum(arr1,arr2)
np.where(arr1>3,arr1,arr2)
arr = np.arange(5)

print(arr.sum())

print(arr.mean())
arr = np.array([[1,2,3],[1,2,5]])

print(arr.sum(axis=1))

print(arr.sum(axis=0))

print(arr.mean(axis=1))

print(arr.mean(axis=0))

arr
arr = np.array([2,3,4,1,5])

arr.sort()

arr
arr = np.array([3,2,4,5,2,3,4])

np.unique(arr)