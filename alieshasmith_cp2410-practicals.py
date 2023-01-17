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
# Practical 01 - Introduction

# Q1

def is_multiple(n,m):

    return n % m == 0



is_multiple(10,5)



# Q2

new_list = [2**i for i in range (0, 9)]



new_list



# Q3

def is_distinct(list):

    for i in range(0, len(list)-1):

        for k in range(i+1, len(list)):

            if (i!=k):

                if(list[i]==list[k]):

                    return False

    return True



notDistinct = [0, 1, 2, 2]

distinct = [0, 1, 2, 3, 4]

is_distinct(notDistinct)

is_distinct(distinct)



# Q4

def harmonic_list_gen(n):

    h = 0

    for i in range(1, n+1):

        h += 1/i

        yield h

        



    

    

        
# Prac 3 



# Q5

import sys

from time import time

from dynamic_array import DynamicArray



try:

    maxN = int(sys.argv[1])

except:

    maxN = 10000000



from time import time  # import time function from time module





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

    

# Q6



import ctypes  # provides low-level arrays





class DynamicArray:

    """A dynamic array class akin to a simplified Python list."""



    def __init__(self, resize_factor):

        """Create an empty array."""

        self._n = 0  # count actual elements

        self._capacity = 1  # default array capacity

        self._A = self._make_array(self._capacity)  # low-level array

        self.resize_factor = resize_factor



    def __len__(self):

        """Return number of elements stored in the array."""

        return self._n



    def __getitem__(self, k):

        """Return element at index k."""

        if not 0 <= k < self._n:

            raise IndexError('invalid index')

        return self._A[k]  # retrieve from array



    def append(self, obj):

        """Add object to end of the array."""

        if self._n == self._capacity:  # not enough room

            self._resize(self.resize_factor * self._capacity)  # so double capacity

        self._A[self._n] = obj

        self._n += 1



    def _resize(self, c):  # nonpublic utitity

        """Resize internal array to capacity c."""

        B = self._make_array(c)  # new (bigger) array

        for k in range(self._n):  # for each existing value

            B[k] = self._A[k]

        self._A = B  # use the bigger array

        self._capacity = c



    def _make_array(self, c):  # nonpublic utitity

        """Return new array with capacity c."""

        return (c * ctypes.py_object)()  # see ctypes documentation



    def insert(self, k, value):

        """Insert value at index k, shifting subsequent values rightward."""

        # (for simplicity, we assume 0 <= k <= n in this verion)

        if self._n == self._capacity:  # not enough room

            self._resize(self.resize_factor * self._capacity)  # so double capacity

        for j in range(self._n, k, -1):  # shift rightmost first

            self._A[j] = self._A[j - 1]

        self._A[k] = value  # store newest element

        self._n += 1



    def remove(self, value):

        """Remove first occurrence of value (or raise ValueError)."""

        # note: we do not consider shrinking the dynamic array in this version

        for k in range(self._n):

            if self._A[k] == value:  # found a match!

                for j in range(k, self._n - 1):  # shift others to fill gap

                    self._A[j] = self._A[j + 1]

                self._A[self._n - 1] = None  # help garbage collection

                self._n -= 1  # we have one less item

                return  # exit immediately

        raise ValueError('value not found')  # only reached if no match
