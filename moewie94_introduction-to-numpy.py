!pip install numpy
import numpy as np
c = np.array(1, dtype=np.float32)

print(f"c: shape {c.shape} - dtype: {c.dtype}")

print(c)
v = np.array([1.0, 2.0, 3.0], dtype=np.int32)

print(f"v: shape {v.shape} - dtype: {v.dtype}")

print(v)
A = np.array([[1, 2, 3, 4], [3, 4, 5, 6], [6, 7, 8, 9]])

print(f"A: shape {A.shape} - dtype: {A.dtype}")

print(A)
data = np.array([[1, 2], [4, 5]])

print(data)
data = np.zeros(5)

print(data)
data = np.ones((3, 3))

print(data)
data = np.arange(5)

print(data)
arr = np.array([1, 2, 3])

print(f"Original data type: {arr.dtype}")
arr = np.float32(arr)

print(f"Converted data type: {arr.dtype}")
arr = arr.astype(np.float64)

print(f"Converted data type: {arr.dtype}")
arr = np.arange(10)

print(arr)
print(arr[1])
print(arr[-3])
print(arr[[0, 1]])
print(arr[5:8])
print(arr[:3])
print(arr[3:])
print(arr[:])
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"Get a row: {arr[0]} or {arr[0, :]}")
print(f"Get a col: {arr[:, 1]}")
print(f"Get an element: {arr[0, 1]} or {arr[0][1]}")
print(f"Indexing with slices: {arr[:2, 1:]}")
arr = np.empty((3, 3, 3))

print(arr)
print(f"Get element indexed 0 in the 2nd dimension:")

print(arr[:, 0, :])

print("---")

print(f"Get element indexed 0 in the first dimension:")

print(arr[0, :, :])

print("or")

print(arr[0, ...])

print("---")

print(f"Get element indexed 0 in the last dimension:")

print(arr[:, :, 0])

print("or")

print(arr[..., 0])

print("---")
ids = np.array([1, 2, 3])

names = np.array(["a", "b", "a"])
print(ids[names == "a"])
arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

print(f"addition: {arr1 + arr2}")

print("---")

print(f"compare: {arr1 > arr2}")
arr = np.array([1, 2, 3])

scalar = 3



print(f"addition: {arr+scalar}")

print("---")

print(f"multiplication: {arr*scalar}")

print("---")

print(f"power: {arr**scalar}")

print("---")

print(f"compare: {arr == scalar}")
arr1 = np.arange(5)

arr2 = 5 * np.random.rand(5)

print(f"original arrays:")

print(arr1)

print("and")

print(arr2)

print("---")



print(f"square root: {np.sqrt(arr1)}")

print("---")

print(f"exponential root: {np.exp(arr1)}")

print("---")

print(f"max: {np.maximum(arr1, arr2)}")

print("---")
arr = np.arange(10)

print(f"original shape: {arr.shape}")

print(f"original array: {arr}")



arr = arr.reshape((2, 5))

print(f"new shape: {arr.shape}")

print(f"new array: {arr}")
arr = np.arange(10)

arr = arr.reshape((2, 5))

print(f"original array: {arr}")



print(f"transposed array: {arr.T} or {arr.transpose((1, 0))}")



print(f"swap two axes: {arr.swapaxes(1, 0)}")
mat1 = np.random.randint(low=0, high=10, size=(3, 3))

mat2 = np.random.randint(low=0, high=10, size=(3, 3))

print(f"matrix 1: {mat1}")

print(f"matrix 2: {mat2}")

print("---")



print(f"concatenated matrix (along axis 0): {np.concatenate([mat1, mat2], axis=0)}")

print(f"concatenated matrix (along axis 1): {np.concatenate([mat1, mat2], axis=1)}")
import matplotlib.pyplot as plt



# generate random numbers

X = np.random.randint(low=0, high=5, size=(30, 2))



# visualize generated numbers

plt.plot(X[:, 0], X[:, 1], "ro", alpha=0.5)

plt.show()
import matplotlib.pyplot as plt

import seaborn as sns



# generate random numbers

means = [[2, 2]]

cov = [[3, 0], [0, 1]]

N = 1000

X0 = np.random.multivariate_normal(means[0], cov, N)



# visualize generated numbers

fig, ax = plt.subplots(1, 2, figsize=(10, 5))



ax[0].plot(X0[:, 0], X0[:, 1], "ro", alpha=0.2)

ax[0].set_title("Random 2D points")

ax[0].set_xlabel("X")

ax[0].set_ylabel("Y")



ax[1] = sns.kdeplot(X0[:, 0], X0[:, 1], shade=True)

ax[1].set_title("Underlying Gaussian")

ax[1].set_xlabel("X")

ax[1].set_ylabel("Y")



plt.show()
print(np.random.permutation(10))
import matplotlib.pyplot as plt

import seaborn as sns



# generate random numbers

X = np.random.randn(500, 1)



# visualize generated numbers

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].plot(X[:, 0], [7e-3]*500, "ro", alpha=0.1)

ax[0].set_title("Generated data points")

ax[0].set_xlim([-3, 3])



ax[1] = sns.kdeplot(X[:, 0], shade=True)

ax[1].set_title("Underlying Gaussian")

ax[1].set_xlim([-3, 3])

plt.show()
import matplotlib.pyplot as plt

import seaborn as sns



# generate random numbers

X = np.random.uniform(low=0, high=10, size=(100, 1))



# visualize generated numbers

plt.plot(X[:, 0], [7e-3]*100, "ro", alpha=0.5)

plt.xlim([0, 10])

plt.show()
arr = np.random.randint(low=0, high=10, size=(2, 2))

print(arr)
print(f"sum: {arr.sum()} or {np.sum(arr)}")

print(f"sum along axis 0: {arr.sum(axis=0)} or {np.sum(arr, axis=0)}")

print(f"argmin: {arr.argmin()} or {np.argmin(arr)}")

print(f"cumsum: {arr.cumsum()} or {np.cumsum(arr)}")
arr = np.random.randn(300)
import matplotlib.pyplot as plt

import seaborn as sns



# visualize generated numbers

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].plot(arr, [7e-3]*300, "ro", alpha=0.1)

ax[0].set_title("Generated data points")

ax[0].set_xlim([-3, 3])



ax[1] = sns.kdeplot(arr, shade=True)

ax[1].set_title("Underlying Gaussian")

ax[1].set_xlim([-3, 3])

plt.show()
print(f"mean: {arr.mean()} or {np.mean(arr)}")

print(f"std: {arr.std()} or {np.std(arr)}")

print(f"var: {arr.var()} or {np.var(arr)}")
mat1 = np.random.randint(low=0, high=10, size=(3, 3))

print(f"original matrix:")

print(mat1)

print("---")



mat1.sort()

print(f"sorted matrix:")

print(mat1)

print("---")



mat1.sort(axis=0)

print(f"sorted along axis 0")

print(mat1)

print("---")
mat1 = np.random.randint(low=0, high=10, size=10)

mat2 = np.random.randint(low=0, high=10, size=8)

print(f"matrix 1: {mat1}")

print(f"matrix 2: {mat2}")

print("---")



print(f"unique elements of mat1: {np.unique(mat1)}")

print("---")

print(f"intersection of mat1 and mat2: {np.intersect1d(mat1, mat2)}")

print("---")

print(f"union of mat1 and mat2: {np.union1d(mat1, mat2)}")

print("---")

print(f"whether each element of mat1 is in mat2 or not: {np.in1d(mat1, mat2)}")

print("---")

print(f"elements in mat1 but not in mat2: {np.setdiff1d(mat1, mat2)}")

print("---")
mat1 = np.random.randint(low=0, high=10, size=(2, 2))

mat2 = np.random.randint(low=0, high=10, size=(2, 2))

print(f"matrix 1:")

print(mat1)

print(f"matrix 2:")

print(mat2)

print("---")



print(f"matrix multiplication")

print(np.dot(mat1, mat2))

print("---")

print(f"matrix trace")

print(np.trace(mat1))

print("---")

print(f"matrix determinant")

print(np.linalg.det(mat1))

print("---")

print("---")

print(f"matrix pseudo-inverse")

print(np.linalg.pinv(mat1))

print("---")