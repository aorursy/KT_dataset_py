# Question 4. 



# " product of two positive integers, m and n."



# Say m = 6 and n = 3

# The product of 6 and 3 = 18

# 6 * 3 = 6 + 6 + 6

# Say n = 4, we simply add another 6 to 6 * 3

# 6 * 4 = 6 + 6 + 6 + 6



# Simply, 6 * n = 6 + ( 6 * (n - 1))

# m * n = m + (m * (n - 1))



def product(m,n):

    if n == 1: 

        return m

    else:

        return m + product(m, (n-1))



product(6, 3)



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
# Question 6



class ResizeableDynamicArray(DynamicArray):

    def __init__(self, resize_factor):

        super().__init__()

        self.resize_factor = resize_factor

    

    def append(self, obj):

        """Add object to end of the array."""

        if self._n == self._capacity:                 

              self._resize(int(self.resize_factor * self._capacity) + 1)            

        self._A[self._n] = obj

        self._n += 1



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
