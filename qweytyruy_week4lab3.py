# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import sys

from time import time

from dynamic_array import DynamicArray



# Question 1

# max(sequence, start_point)

# if the sequence == start_position

# return sequence[start]

# else max_rest = max(sequence, start_position + 1)

# if sequence[start_position] > max_rest then return sequence[start_position]

# else return max_rest





# Question 2

def power(x, n):

    """Compute the value x**n for integer n."""

    if n == 0:

        return 1

    else:

        return x * power(x, n - 1)

# power(2,5): return 2 * A

# A = power(2,4): return 2 * B

# B = power(2,3): return 2 * C

# C = power(2,2): return 2 * D

# D = power(2,1): return 2 * E

# E = power(2,0): return 1



# power(2,5) = 32





# Question 3

def power2(x, n):

    """Compute the value x**n for integer n."""

    if n == 0:

        return 1

    else:

        partial = power(x, n // 2) # rely on truncated division

    result = partial * partial

    if n % 2 == 1: # if n odd, include extra factor of x

        result *= x

    return result



# power(2,18): return 2 * A

# A = power(2,9): return 2 * B

# B = power(2,4): return 2 * C

# C = power(2,2): return 2 * D

# D = power(2,1): return 2 * E

# E = power(2,0): return 1



# power(2,18) = 262144





# Question 4

# Multiplication is essentially repeated addition and can be generalised by the

# recursive algorithm: m*n = m + (m*(n-1))

# multiply(m,n)

# if n == 1 then base case m*1 = m

# return m

# else

# return m + multiply(m,n-1)





# Question 5

try:

    maxN = int(sys.argv[1])

except:

    maxN = 10000000

from time import time # import time function from time module





def compute_average(n):

    """Perform n appends to an empty list and return average time elapsed."""

    data = DynamicArray()

    start = time()  # record the start time (in seconds)

    for k in range(n):

        data.append(None)

    end = time()  # record the end time (in seconds)

    return (end - start) / n  # compute average per operation



n = 10

while n <= maxN:

    print('Average of {0:.3f} for n {1}'.format(compute_average(n) * 1000000, n))

    n *= 10





# Question 6

class DynamicArrayWithResizeFactor(DynamicArray):

    """ A dynamic array class which allows for a custom resize factor. """

    def __init__(self, resize_factor):

        super().__init__()

        self._resize_factor = resize_factor

    def append(self, obj):

        """ Modified version of append to use the resize factor. """

        if self._n == self._capacity: # not enough room

 # cast to int to make allow for fractional resize_factors, add 1 to make sure

 # capacity always increases

            self._resize(int(self._resize_factor * self._capacity) + 1)

        self._A[self._n] = obj

        self._n += 1


