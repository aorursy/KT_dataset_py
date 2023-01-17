import numpy as np
# check the version

print(np.__version__)
# integer array

print(np.array([1, 4, 2, 5, 3]))
# upcast

print(np.array([3.14, 4, 2, 3]))
# assign data type explicitely

np.array([1, 2, 3, 4], dtype='float32')
# initialize a multidimensional array using a list of lists

print(np.array([range(i, i + 3) for i in [2, 4, 6]]))
# Create a length-10 integer array filled with zeros

np.zeros(10, dtype='int16')
# 2nd way to assign dtype

np.zeros(10, dtype=np.int16)
# Create a 3x5 floating-point array filled with 1s

np.ones((3, 5), dtype='float')
# Create a 3x5 array filled with 3.14

print(np.full((3, 5), 3.14))
# Create an array filled with a linear sequence

print(np.arange(0, 20, 2))
# Create an array of five values evenly spaced between 0 and 1

print(np.linspace(0, 1, 5))
# Create a 3x3 array of uniformly distributed random values between 0 and 1

print(np.random.random((3, 3)))
# Create a 3x3 array of normally distributed random values with mean 0 and standard deviation 1

print(np.random.normal(0, 1, (3, 3)))
# Create a 3x3 array of random integers in the interval [0, 10)

print(np.random.randint(0, 10, (3, 3)))
# identity matrix

print(np.eye(3))
# Create an uninitialized array of three integers

print(np.empty(3))
np.random.seed(0)  # seed for reproducibility

x1 = np.random.randint(10, size=6) # One-dimensional array

x2 = np.random.randint(10, size=(3, 4)) # Two-dimensional array

x3 = np.random.randint(10, size=(3, 4, 5)) # Three-dimensional array
print('x1\n', x1)

print('x2\n', x2)

print('x3\n', x3)
print("x3 ndim: ", x3.ndim)

print("x3 shape:", x3.shape)

print("x3 size: ", x3.size)
print("dtype:", x3.dtype)
print("itemsize:", x3.itemsize, "bytes")

print("nbytes:", x3.nbytes, "bytes")
print('type of x1[0] :', type(x1[0]))

x1[0] = 3.14159

print('Take care because the value is truncated :', x1[0])
x = np.arange(10)

print(x)
print(x[::-1]) # all elements, reversed
print(x[5::-2]) # reversed every other from index 5
print(x2)
print(x2[::-1, ::-1]) # subarray dimensions can even be reversed together
print(x2[:, 0]) # first column of x2
print(x2[0, :]) # first row of x2

print(x2[0]) # simplied version
print(x2)
x2_sub = x2[:2, :2]

print(x2_sub)
x2_sub[0, 0] = 99

print(x2_sub)
print(x2) # note that the original [0,0] has been changed too.
x2_sub_copy = x2[:2, :2].copy()

print(x2_sub_copy)
x2_sub_copy[0, 0] = 42

print(x2_sub_copy)
print(x2) # the original [0,0] hasn't been changed.
# put the numbers 1 through 9 in a 3×3 grid

grid = np.arange(1, 10).reshape((3, 3))

print(grid)
x = np.array([1, 2, 3])

print(x)

print(x.shape)
# method 1 : row vector with the reshape method

row_reshape1 = x.reshape((1, 3))

print(row_reshape1)

print(row_reshape1.shape)
# method 2 : row vector with newaxis keyword within a slice operation

row_reshape2 = x[np.newaxis, :]

print(row_reshape2)

print(row_reshape2.shape)
# method 1 : column vector with the reshape method

col_reshape1 = x.reshape((3, 1))

print(col_reshape1)

print(col_reshape1.shape)
# method 2 : column vector with newaxis keyword within a slice operation

col_reshape2 = x[:, np.newaxis]

print(col_reshape2)

print(col_reshape2.shape)
x = np.array([1, 2, 3])

y = np.array([3, 2, 1])

z = [99, 99, 99]

print(np.concatenate([x, y, z]))
grid = np.array([[1, 2, 3],

                 [4, 5, 6]])

print(np.concatenate([grid, grid]))
# concatenate along the first axis (zero-indexed)

print(np.concatenate([grid, grid], axis=0))
# concatenate along the second axis (zero-indexed)

print(np.concatenate([grid, grid], axis=1))
x = np.array([1, 2, 3])

grid = np.array([[9, 8, 7],

                 [6, 5, 4]])



# vertically stack the arrays

print(np.vstack([x, grid]))
# horizontally stack the arrays

y = np.array([[99],

              [99]])

print(np.hstack([grid, y]))
# np.dstack will stack arrays along the third axis.
x = [1, 2, 3, 99, 99, 3, 2, 1]

x1, x2, x3 = np.split(x, [3, 5])

print(x1, x2, x3)
grid = np.arange(16).reshape((4, 4))

print(grid)
# Split an array into multiple sub-arrays vertically (row-wise).

upper, lower = np.vsplit(grid, [2])

print(upper)

print(lower)
# Split an array into multiple sub-arrays horizontally (column-wise).

left, right = np.hsplit(grid, [2])

print(left)

print(right)
x = np.arange(4)

print("x           =", x)

print("x + 5       =", x + 5)

print('np.add(x,5) =', np.add(x,5)) # the + operator is a wrapper for the add function
# Absolute value

x = np.array([-2, -1, 0, 1, 2])

print(np.absolute(x))

print(np.abs(x))
x = np.array([3 - 4j, 4 - 3j, 2 + 0j, 0 + 1j])

print(np.abs(x))
# Another excellent source for more specialized and obscure ufuncs is the submodule scipy.special.

from scipy import special
# Gamma functions (generalized factorials) and related functions

x = [1, 5, 10]



print("gamma(x)     =", special.gamma(x))

print("ln|gamma(x)| =", special.gammaln(x))

print("beta(x, 2)   =", special.beta(x, 2))
# Error function (integral of Gaussian)

# its complement, and its inverse

x = np.array([0, 0.3, 0.7, 1.0])

print("erf(x)  =", special.erf(x))

print("erfc(x) =", special.erfc(x))

print("erfinv(x) =", special.erfinv(x))
x = np.arange(5)

y = np.empty(5)

np.multiply(x, 10, out=y)

print(y)
y = np.zeros(10)

np.power(2, x, out=y[::2])

print(x)

print(y)
# sum of all elements in the array

x = np.arange(1, 6)

print(np.add.reduce(x))
# store all the intermediate results of the computation

print(x)

print(np.add.accumulate(x))
x = np.arange(1, 6)

print(np.add.outer(x, x))
# this is universal outer function

print(np.multiply.outer(x, x))
# this is numpy.outer() function

print(np.outer(x, x))
big_array = np.random.rand(1000)

%timeit sum(big_array)

%timeit np.sum(big_array)
np.min(big_array), np.max(big_array)
# shorter syntax

print(big_array.min(), big_array.max(), big_array.sum())
M = np.random.random((3, 4))

print(M)
# we can find the minimum value within each column by specifying axis=0

print(M.min(axis=0))
# we can find the maximum value within each row

print(M.max(axis=1))
import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns; 

sns.set()  

%matplotlib inline
import os

print(os.listdir("../input"))
data = pd.read_csv('../input/numpytutorial/president_heights.csv')

heights = np.array(data['height(cm)'])

print(heights)
print("Mean height:       ", heights.mean())

print("Standard deviation:", heights.std())

print("Minimum height:    ", heights.min())

print("Maximum height:    ", heights.max())
print("25th percentile:   ", np.percentile(heights, 25))

print("Median:            ", np.median(heights))

print("75th percentile:   ", np.percentile(heights, 75))
plt.hist(heights)

plt.title('Height Distribution of US Presidents')

plt.xlabel('height (cm)')

plt.ylabel('number');
# normal way : element-by-element

a = np.array([0, 1, 2])

b = np.array([5, 5, 5])

print(a + b)
# broadcasting way

print(a + 5)
M = np.ones((3, 3))

print('M\n', M)

print('a\n', a)

print('M+a\n', M + a)
a = np.arange(3)

b = np.arange(3)[:, np.newaxis]



print(a)

print(b)
print(a + b)
M = np.ones((2, 3))

a = np.arange(3)
print(M + a)
a = np.arange(3).reshape((3, 1))

b = np.arange(3)
print(a + b)
X = np.random.random((10, 3))

print(X)
Xmean = X.mean(0)

print(Xmean)
X_centered = X - Xmean

print(X_centered)
print(X_centered.mean(0))
x = np.linspace(0, 5, 50)

y = np.linspace(0, 5, 50)[:, np.newaxis]

print(x.shape)

print(y.shape)
z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

print(z.shape)
# imshow : Display an image, i.e. data on a 2D regular raster.

# origin : Place the [0,0] index of the array in the upper left or lower left corner of the axes.

# extent : scalars (left, right, bottom, top), the bounding box in data coordinates that the image will fill.

# cmap : str or Colormap, the Colormap instance or registered colormap name used to map scalar data to colors. 

# viridis : perceptually uniform shades of blue-green-yellow

plt.imshow(z, origin='lower', extent=[0, 5, 0, 5], cmap='viridis')

plt.colorbar();

rainfall = pd.read_csv('../input/numpytutorialseattle/Seattle2014.csv')['PRCP'].values

inches = rainfall / 254  # 1/10mm -> inches

inches.shape
plt.hist(inches, 40);
x = np.array([1, 2, 3, 4, 5])
print(x < 3)
print('2 * x  :', 2 * x)

print('x ** 2 :', x ** 2)

print((2 * x) == (x ** 2))
rng = np.random.RandomState(0)

x = rng.randint(10, size=(3, 4))

print(x)
print(x < 6)
print(x)
# To count the number of True entries in a Boolean array

np.count_nonzero(x < 6)
np.sum(x < 6)
# how many values less than 6 in each row?

print(np.sum(x < 6, axis=1))
# how many values less than 6 in each column?

print(np.sum(x < 6, axis=0))
# are there any values greater than 8?

np.any(x > 8)
# are all values less than 10?

np.all(x < 10)
# are all values in each row less than 8?

print(np.all(x < 8, axis=1))
np.sum((inches > 0.5) & (inches < 1))
print("Number days without rain:      ", np.sum(inches == 0))

print("Number days with rain:         ", np.sum(inches != 0))

print("Days with more than 0.5 inches:", np.sum(inches > 0.5))

print("Rainy days with < 0.1 inches  :", np.sum((inches > 0) & (inches < 0.2)))
print(x)

print(x < 5)
print(x[x < 5])
bool(42), bool(0)
bool(42 and 0)
A = np.array([1, 0, 1, 0, 1, 0], dtype=bool)

B = np.array([1, 1, 1, 0, 1, 1], dtype=bool)

print(A | B)
rand = np.random.RandomState(42)

x = rand.randint(100, size=10)

print(x)
ind = [3, 7, 4, 5]

print(x[ind])
ind = np.array([[3, 7],

                [4, 5]])

print(x[ind])
X = np.arange(12).reshape((3, 4))

print(X)
row = np.array([0, 1, 2])

col = np.array([2, 1, 3])

print(X[row, col])

# Notice that the first value in the result is X[0, 2], the second is X[1, 1], and the third is X[2, 3].
print(X[row[:, np.newaxis], col])
print(X)
print(X[2, [2, 0, 1]])
print(X[1:, [2, 0, 1]])
mask = np.array([1, 0, 1, 0], dtype=bool)

print(row)

print()

print(X[row[:, np.newaxis], mask])
mean = [0, 0]

cov = [[1, 2],

       [2, 5]]

X = rand.multivariate_normal(mean, cov, 100) # Draw random samples from a multivariate normal distribution.

X.shape
plt.scatter(X[:, 0], X[:, 1]);
indices = np.random.choice(X.shape[0], 20, replace=False)

print(indices)

print(indices.shape)
selection = X[indices]  # fancy indexing here

print(selection.shape)
plt.scatter(X[:, 0], X[:, 1], alpha=0.3)

plt.scatter(selection[:, 0], selection[:, 1], facecolor='red', s=50);
x = np.zeros(10)

i = [2, 3, 3, 4, 4, 4]

x[i] += 1

print(x)

x[i] += 1

print(x)
x = np.zeros(10)

i = [2, 3, 3, 4, 4, 4]

np.add.at(x, i, 1)

print(x)
np.random.seed(42)

x = np.random.randn(100)

print('x\n', x.shape)



# compute a histogram by hand

bins = np.linspace(-5, 5, 20)

print('bins\n', bins.shape)

counts = np.zeros_like(bins) # Return an array of zeros with the same shape and type as a given array.

print('counts\n', counts)



# find the appropriate bin for each x

i = np.searchsorted(bins, x) 

# Find indices where elements should be inserted to maintain order.

# Find the indices into a sorted array a such that, 

# if the corresponding elements in v were inserted before the indices, the order of a would be preserved.

print('i.shape\n', i.shape)

print('i\n', i)



# add 1 to each of these bins

np.add.at(counts, i, 1)

print('counts\n', counts)
# plot the results

plt.plot(bins, counts, linestyle='steps');
# the same can be realized by calling the function hist()

plt.hist(x, bins, histtype='step');
x = np.array([2, 1, 4, 3, 5])

print('sorted   :', np.sort(x))

print('original :', x)
x = np.array([2, 1, 4, 3, 5])

i = np.argsort(x)

print(i)

print(x[i])
X = np.array([[5,3,6],

              [1,0,2],

              [9,7,8]]) 
# sort each column of X

print(np.sort(X, axis=0))
# sort each row of X

print(np.sort(X, axis=1))
# {8.2} Partial Sorts: Partitioning
x = np.array([7, 2, 3, 1, 6, 5, 4])

print(np.partition(x, 3))
rand = np.random.RandomState(18)

X = rand.randint(0, 10, (4, 6))

print(X)

print()

# the first two slots in each row contain the smallest values from that row

print(np.partition(X, 2, axis=1)) 
X = rand.rand(10, 2)

print(X)
plt.scatter(X[:, 0], X[:, 1], s=100);
# for each pair of points, compute differences in their coordinates

differences = X[:, np.newaxis, :] - X[np.newaxis, :, :]

print(differences.shape)

#print(differences)
# square the coordinate differences

sq_differences = differences ** 2

print(sq_differences.shape)
# sum the coordinate differences to get the squared distance

dist_sq = sq_differences.sum(-1)

dist_sq.shape
print(dist_sq.diagonal())
nearest = np.argsort(dist_sq, axis=1)

print(nearest)
K = 2

nearest_partition = np.argpartition(dist_sq, K + 1, axis=1)
plt.scatter(X[:, 0], X[:, 1], s=100)



# draw lines from each point to its two nearest neighbors

K = 2



for i in range(X.shape[0]):

    for j in nearest_partition[i, :K+1]:

        # plot a line from X[i] to X[j]

        # use some zip magic to make it happen:

        plt.plot(*zip(X[j], X[i]), color='black')