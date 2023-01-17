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
def insertion_sort(q):

    """Sort list of comparable elements into nondecreasing order.

    Iterates from index 1 backwards through list"""

    for i in range(1, len(q)):

        current = q[i]

        pos = i

        while pos > 0 and q[pos - 1] > current:

            q[j] = q[pos - 1]  # 

            pos -= 1

        q[pos] = current
def selection_sort(q):

    

    for i in range(len(q)):

        

        min_i = i

        for j in range(i+1, len(q)):  # Increment index to smallest value in list

            if q[min_i] > q[j]:

                min_i = j

        # Swap ith el with min el pythonically

        q[i], q[min_i] = q[min_i], q[i]

        # Else need temp variable

        


class PriorityQueueBase:

  """Abstract base class for a priority queue."""



  #------------------------------ nested _Item class ------------------------------

  class _Item:

    """Lightweight composite to store priority queue items."""

    __slots__ = '_key', '_value'



    def __init__(self, k, v):

      self._key = k

      self._value = v



    def __lt__(self, other):

      return self._key < other._key    # compare items based on their keys



    def __repr__(self):

      return '({0},{1})'.format(self._key, self._value)



  #------------------------------ public behaviors ------------------------------

  def is_empty(self):                  # concrete method assuming abstract len

    """Return True if the priority queue is empty."""

    return len(self) == 0



  def __len__(self):

    """Return the number of items in the priority queue."""

    raise NotImplementedError('must be implemented by subclass')



  def add(self, key, value):

    """Add a key-value pair."""

    raise NotImplementedError('must be implemented by subclass')



  def min(self):

    """Return but do not remove (k,v) tuple with minimum key.



    Raise Empty exception if empty.

    """

    raise NotImplementedError('must be implemented by subclass')



  def remove_min(self):

    """Remove and return (k,v) tuple with minimum key.



    Raise Empty exception if empty.

    """

    raise NotImplementedError('must be implemented by subclass')



###    

### HEAP PRIORITY QUEUE ###    

###



class HeapPriorityQueue(PriorityQueueBase): # base class defines _Item

  """A min-oriented priority queue implemented with a binary heap."""



  #------------------------------ nonpublic behaviors ------------------------------

  def _parent(self, j):

    return (j-1) // 2



  def _left(self, j):

    return 2*j + 1

  

  def _right(self, j):

    return 2*j + 2



  def _has_left(self, j):

    return self._left(j) < len(self._data)     # index beyond end of list?

  

  def _has_right(self, j):

    return self._right(j) < len(self._data)    # index beyond end of list?

  

  def _swap(self, i, j):

    """Swap the elements at indices i and j of array."""

    self._data[i], self._data[j] = self._data[j], self._data[i]



  def _upheap(self, j):

    parent = self._parent(j)

    if j > 0 and self._data[j] < self._data[parent]:

      self._swap(j, parent)

      self._upheap(parent)             # recur at position of parent

  

  def _downheap(self, j):

    if self._has_left(j):

      left = self._left(j)

      small_child = left               # although right may be smaller

      if self._has_right(j):

        right = self._right(j)

        if self._data[right] < self._data[left]:

          small_child = right

      if self._data[small_child] < self._data[j]:

        self._swap(j, small_child)

        self._downheap(small_child)    # recur at position of small child



  #------------------------------ public behaviors ------------------------------

  def __init__(self):

    """Create a new empty Priority Queue."""

    self._data = []



  def __len__(self):

    """Return the number of items in the priority queue."""

    return len(self._data)



  def add(self, key, value):

    """Add a key-value pair to the priority queue."""

    self._data.append(self._Item(key, value))

    self._upheap(len(self._data) - 1)            # upheap newly added position

  

  def min(self):

    """Return but do not remove (k,v) tuple with minimum key.



    Raise Empty exception if empty.

    """

    if self.is_empty():

      raise Exception('Priority queue is empty.')

    item = self._data[0]

    return (item._key, item._value)



  def remove_min(self):

    """Remove and return (k,v) tuple with minimum key.



    Raise Empty exception if empty.

    """

    if self.is_empty():

      raise Exception('Priority queue is empty.')

    self._swap(0, len(self._data) - 1)           # put minimum item at the end

    item = self._data.pop()                      # and remove it from the list;

    self._downheap(0)                            # then fix new root

    return (item._key, item._value)



    
heap = HeapPriorityQueue()



lkeys = [5, 1, 4, 7, 3, 9, 0, 2, 8]

val = 0

for d, key in enumerate(lkeys):

    heap.add(key, val)

    

print(heap._data)