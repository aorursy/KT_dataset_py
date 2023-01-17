# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def find_max(s):

    if len(s) < 1:

        return -1

    if len(s) <= 1: # Recursively iterates through list until all thats left is the largest item

        return s[0] # max is item

    if s[0] > s[-1]: # if first item is larger

        return find_max(s[:-1]) # recursive call will compare first item against second last item

    else: # else last item larger

        return find_max(s[1:]) # implicitly compare second item against last item

    

    

print(find_max([10,2,3,9,4,7,3]))

print(find_max([1,5,146,6,8,145,147,145]))
def power(x, n):

    """Compute the value x**n for integer n"""

    if n == 0:

        return 1

    else:

        return x * power(x, n - 1)
def power(x, n):

    """Compute the value x**n for integer n"""

    if n == 0:

        return 1

    else:

        partial = power(x, n // 2) # rely on truncated div

        result = partial * partial

        if n % 2 == 1: # if odd n, include extra factor of x

            result *= x

        return result

power(2, 18)



def additive_product(m, n):

    

    if n == 0: # Base case

        return 0

    elif n < 0 and m > 0: # Negative multiplier

        return -m + additive_product(m, n+1)

    elif n > 0 and m < 0: # Negative base

        return m + additive_product(m, n-1)

    else:

        return abs(m) + additive_product(abs(m), abs(n)-1)



print(additive_product(5, 2))

print(additive_product(-5, 2))

print(additive_product(5, -2))

print(additive_product(-5, -2))
import ctypes                                      # provides low-level arrays



class DynamicArray:

  """A dynamic array class akin to a simplified Python list."""



  def __init__(self):

    """Create an empty array."""

    self._n = 0                                    # count actual elements

    self._capacity = 1                             # default array capacity

    self._A = self._make_array(self._capacity)     # low-level array

    

  def __len__(self):

    """Return number of elements stored in the array."""

    return self._n

    

  def __getitem__(self, k):

    """Return element at index k."""

    if not 0 <= k < self._n:

      raise IndexError('invalid index')

    return self._A[k]                              # retrieve from array

  

  def append(self, obj):

    """Add object to end of the array."""

    if self._n == self._capacity:                  # not enough room

      self._resize(2 * self._capacity)             # so double capacity

    self._A[self._n] = obj

    self._n += 1



  def _resize(self, c):                            # nonpublic utitity

    """Resize internal array to capacity c."""

    B = self._make_array(c)                        # new (bigger) array

    for k in range(self._n):                       # for each existing value

      B[k] = self._A[k]

    self._A = B                                    # use the bigger array

    self._capacity = c



  def _make_array(self, c):                        # nonpublic utitity

     """Return new array with capacity c."""   

     return (c * ctypes.py_object)()               # see ctypes documentation



  def insert(self, k, value):

    """Insert value at index k, shifting subsequent values rightward."""

    # (for simplicity, we assume 0 <= k <= n in this verion)

    if self._n == self._capacity:                  # not enough room

      self._resize(2 * self._capacity)             # so double capacity

    for j in range(self._n, k, -1):                # shift rightmost first

      self._A[j] = self._A[j-1]

    self._A[k] = value                             # store newest element

    self._n += 1



  def remove(self, value):

    """Remove first occurrence of value (or raise ValueError)."""

    # note: we do not consider shrinking the dynamic array in this version

    for k in range(self._n):

      if self._A[k] == value:              # found a match!

        for j in range(k, self._n - 1):    # shift others to fill gap

          self._A[j] = self._A[j+1]

        self._A[self._n - 1] = None        # help garbage collection

        self._n -= 1                       # we have one less item

        return                             # exit immediately

    raise ValueError('value not found')    # only reached if no match



import sys

from time import time



try:

    maxN = int(sys.argv[1])

except:

    maxN = 10000000



from time import time            # import time function from time module

def compute_average(n):

  """Perform n appends to an empty list and return average time elapsed."""

  data = []

  start = time()                 # record the start time (in seconds)

  for k in range(n):

    data.append(None)

  end = time()                   # record the end time (in seconds)

  return (end - start) / n       # compute average per operation



n = 10

while n <= maxN:

  print('Average of {0:.3f} for n {1}'.format(compute_average(n)*1000000, n))

  n *= 10



import sys

from time import time

try:

    maxN = int(sys.argv[1])

except:

    maxN = 10000000



from time import time            # import time function from time module

def compute_average(n):

  """Perform n appends to an empty Dynamicarray() and return average time elapsed."""

  data = DynamicArray()

  start = time()                 # record the start time (in seconds)

  for k in range(n):

    data.append(None)

  end = time()                   # record the end time (in seconds)

  return (end - start) / n       # compute average per operation



n = 10

while n <= maxN:

  print('Average of {0:.3f} for n {1}'.format(compute_average(n)*1000000, n))

  n *= 10

import ctypes                                      # provides low-level arrays



class DynamicArray:

  """A dynamic array class akin to a simplified Python list."""



  def __init__(self, resize_factor=2):

    """Create an empty array."""

    self._n = 0                                    # count actual elements

    self._capacity = 1                             # default array capacity

    self._resize_factor = resize_factor

    self._A = self._make_array(self._capacity)     # low-level array

    

  def __len__(self):

    """Return number of elements stored in the array."""

    return self._n

    

  def __getitem__(self, k):

    """Return element at index k."""

    if not 0 <= k < self._n:

      raise IndexError('invalid index')

    return self._A[k]                              # retrieve from array

  

  def append(self, obj):

    """Add object to end of the array."""

    if self._n == self._capacity:                  # not enough room

      self._resize(self._resize_factor * self._capacity)             # so double capacity

    self._A[self._n] = obj

    self._n += 1



  def _resize(self, c):                            # nonpublic utitity

    """Resize internal array to capacity c."""

    B = self._make_array(c)                        # new (bigger) array

    for k in range(self._n):                       # for each existing value

      B[k] = self._A[k]

    self._A = B                                    # use the bigger array

    self._capacity = c



  def _make_array(self, c):                        # nonpublic utitity

     """Return new array with capacity c."""   

     return (c * ctypes.py_object)()               # see ctypes documentation



  def insert(self, k, value):

    """Insert value at index k, shifting subsequent values rightward."""

    # (for simplicity, we assume 0 <= k <= n in this verion)

    if self._n == self._capacity:                  # not enough room

      self._resize(self._resize_factor * self._capacity)             # so double capacity

    for j in range(self._n, k, -1):                # shift rightmost first

      self._A[j] = self._A[j-1]

    self._A[k] = value                             # store newest element

    self._n += 1



  def remove(self, value):

    """Remove first occurrence of value (or raise ValueError)."""

    # note: we do not consider shrinking the dynamic array in this version

    for k in range(self._n):

      if self._A[k] == value:              # found a match!

        for j in range(k, self._n - 1):    # shift others to fill gap

          self._A[j] = self._A[j+1]

        self._A[self._n - 1] = None        # help garbage collection

        self._n -= 1                       # we have one less item

        return                             # exit immediately

    raise ValueError('value not found')    # only reached if no match



import sys

from time import time

try:

    maxN = int(sys.argv[1])

except:

    maxN = 10000000



from time import time            # import time function from time module

def compute_average(n, resize_factor):

  """Perform n appends to an empty Dynamicarray() and return average time elapsed."""

  data = DynamicArray(resize_factor)

  start = time()                 # record the start time (in seconds)

  for k in range(n):

    data.append(None)

  end = time()                   # record the end time (in seconds)

  return (end - start) / n       # compute average per operation



n = 10

for i in range(2, 10, 2):

    print(f"DynamicArray with resize factor of {i}")

    while n <= maxN:

      print('Average of {0:.3f} for n {1}'.format(compute_average(n, resize_factor=i)*1000000, n))

      n *= 10

    n = 10
