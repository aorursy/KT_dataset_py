# Q.1 

n = 0

h = 0

s = [1,2,3,5,6,4,90,40,50]

def maxElement (s,n,h):

    if len(s) == n:

        return s

    else:

        s_value = s[n]

        if s_value > h:

            h = s_value

            print(h)

    if n != len(s):

        maxElement(s, n + 1,h)

            

maxElement(s,n,h)
def power (x,n):

    if n==0:

        return 1

    else:

        return x * power(x, n-1)

power(2,5)
def power (x,n):

    if n==0:

        return 1

    else:

        partial = power(x, n // 2)

        result = partial * partial

        if n % 2 == 1:

            result *= x

        return result



power(2,18)
# Q.4



def product(m,n):

    if m == 1:

        return n

    elif n == 1:

            return m

    else:

        return m + product(m,n-1)

    

product(5,2)







import ctypes                                      # provides low-level arrays



class DynamicArray():

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
#Q.6



import ctypes                                      # provides low-level arrays



import sys

from time import time



class ResizeDynamicArray(DynamicArray):

    def __init__(self, resize_factor):

        super().__init__()

        self.resize_factor = resize_factor

    

    def append(self, obj):

        """Add object to end of the array."""

        if self._n == self._capacity:                  # not enough room

              self._resize(int(self.resize_factor * self._capacity) + 1)             # so double capacity

        self._A[self._n] = obj

        self._n += 1







try:

    maxN = int(sys.argv[1])

except:

    maxN = 10000000



from time import time            # import time function from time module

def compute_average_resize(n, resize_num):

  """Perform n appends to an empty list and return average time elapsed."""

  data = ResizeDynamicArray(resize_num)

  start = time()                 # record the start time (in seconds)

  for k in range(n):

    data.append(None)

  end = time()                   # record the end time (in seconds)

  return (end - start) / n       # compute average per operation





def calc_resize_avg(resize_num):

    n = 10

    print('Resize Num {0}'.format(resize_num))

    while n <= maxN:

      print('Average of {0:.3f} for n {1}'.format(compute_average_resize(n,resize_num)*1000000, n))

      n *= 10



calc_resize_avg(2)

calc_resize_avg(4)

calc_resize_avg(8)
