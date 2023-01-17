# Question 1

def find_max_element(sequence, start = 0):

    if start == len(sequence):

        return sequence[start]

    else:

        next_element = find_max_element(sequence, start + 1)

        if sequence[start] > next_element:

            return sequence[start]

        else:

            return next_element
# Question 4

# Give a recursive algorithm to compute the product of two positive integers, m and n, using only addition and subtraction

def multiply_with_addition(m, n): # m will be the base number, n will be number of times m is added to itself

    if n == 1:

        return m           # m * 1 = m

    else:

        return m + multiply_with_addition(m, n - 1) #Recur until n = 1 
# Required For Future Questions

# Copyright 2013, Michael H. Goldwasser

#

# Developed for use with the book:

#

#    Data Structures and Algorithms in Python

#    Michael T. Goodrich, Roberto Tamassia, and Michael H. Goldwasser

#    John Wiley & Sons, 2013

#

# This program is free software: you can redistribute it and/or modify

# it under the terms of the GNU General Public License as published by

# the Free Software Foundation, either version 3 of the License, or

# (at your option) any later version.

#

# This program is distributed in the hope that it will be useful,

# but WITHOUT ANY WARRANTY; without even the implied warranty of

# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the

# GNU General Public License for more details.

#

# You should have received a copy of the GNU General Public License

# along with this program.  If not, see <http://www.gnu.org/licenses/>.



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

        return self._A[k]               # retrieve from array

      

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

# Question 5

# Modify ch05/experiment_list_append.py to investigate the time taken by append operations for DynamicArray (ch05/dynamic_array.py).



# from dynamic_array import DynamicArray ## Imported Dynamic Array Class

import sys

from time import time



try:

    maxN = int(sys.argv[1])

except:

    maxN = 10000000



from time import time            # import time function from time module

def compute_average(n):

    """Perform n appends to an empty list and return average time elapsed."""

    data = DynamicArray()          ## Replaced [] with New Instance of Dynamic Array Class

    start = time()                 # record the start time (in seconds)

    for k in range(n):

        data.append(None)

        end = time()                   # record the end time (in seconds)

    return (end - start) / n       # compute average per operation



n = 10

while n <= maxN:

    print('Average of {0:.3f} for n {1}'.format(compute_average(n)*1000000, n))

    n *= 10



    

#Averages @ Time of Running:

#Average of 11.516 for n 10

#Average of 1.817 for n 100

#Average of 0.750 for n 1000

#Average of 0.816 for n 10000

#Average of 0.769 for n 100000

#Average of 0.701 for n 1000000

#Average of 0.783 for n 10000000
# Question 6

# Create a modified version of DynamicArray (ch05/dyanmic_array.py) that takes a parameter, resize_factor, which it uses to determine the new size (rather than doubling in the original code - self._resize(2 * self._capacity)). Using different values of resize_factor, examine if and how the average time to append changes.



# NEW CLASS

class ResizeableDynamicArray(DynamicArray):

    def __init__(self, resize_factor):

        super().__init__()

        self.resize_factor = resize_factor

    

    def append(self, obj):

        """Add object to end of the array."""

        if self._n == self._capacity:                  # not enough room

              self._resize(int(self.resize_factor * self._capacity) + 1)             # so double capacity

        self._A[self._n] = obj

        self._n += 1



#Tester

import sys

from time import time



try:

    maxN = int(sys.argv[1])

except:

    maxN = 10000000



from time import time            # import time function from time module

def compute_average_resize(n, resize_amount):

    """Perform n appends to an empty list and return average time elapsed."""

    data = ResizeableDynamicArray(resize_amount)          ## Replaced [] with New Instance of Dynamic Array Class

    start = time()                 # record the start time (in seconds)

    for k in range(n):

        data.append(None)

        end = time()                   # record the end time (in seconds)

    return (end - start) / n       # compute average per operation



#

def calculate_resize_average(resize_amount):

    n = 10

    print('Calculating Average for Resize Amount {0}'.format(resize_amount))

    while n <= maxN:

        print('Average of {0:.3f} for n {1}'.format(compute_average_resize(n, resize_amount)*1000000, n))

        n *= 10



calculate_resize_average(1.1)

calculate_resize_average(2)

calculate_resize_average(4)

calculate_resize_average(8)