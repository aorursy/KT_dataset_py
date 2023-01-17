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
def insertion_sort(A):

    """Sort list of comparable elements into nondecreasing order."""

    for k in range(1, len(A)):  # from 1 to n-1

        cur = A[k]  # current element to be inserted

        j = k  # find correct index j for current

        while j > 0 and A[j - 1] > cur:  # element A[j-1] must be after current

            A[j] = A[j - 1]

            j -= 1

        A[j] = cur  # cur is now in the right place

    return A
mylist = [5, 6, 3, 1, 2, 7, 9, 8]

insertion_sort(mylist)

print(mylist)
def selection_sort(A):

    for i in range(len(A)): 

        min_index = i 

        for j in range(i+1, len(A)): 

            if A[min_index] > A[j]: 

                min_index = j 

        print('Swap:', A[i], 'and', A[min_index])

        A[i], A[min_index] = A[min_index], A[i]
mylist2 = [5, 6, 3, 1, 2, 7, 9, 8]

selection_sort(mylist2)

print(mylist2)
class PriorityQueueBase:

    """Abstract base class for a priority queue."""



    # ------------------------------ nested _Item class ------------------------------

    class _Item:

        """Lightweight composite to store priority queue items."""

        __slots__ = '_key', '_value'



        def __init__(self, k, v):

            self._key = k

            self._value = v



        def __lt__(self, other):

            return self._key < other._key  # compare items based on their keys



        def __repr__(self):

            return '({0},{1})'.format(self._key, self._value)



    # ------------------------------ public behaviors ------------------------------

    def is_empty(self):  # concrete method assuming abstract len

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

class HeapPriorityQueue(PriorityQueueBase):  # base class defines _Item

    """A min-oriented priority queue implemented with a binary heap."""



    # ------------------------------ nonpublic behaviors ------------------------------

    def _parent(self, j):

        return (j - 1) // 2



    def _left(self, j):

        return 2 * j + 1



    def _right(self, j):

        return 2 * j + 2



    def _has_left(self, j):

        return self._left(j) < len(self._data)  # index beyond end of list?



    def _has_right(self, j):

        return self._right(j) < len(self._data)  # index beyond end of list?



    def _swap(self, i, j):

        """Swap the elements at indices i and j of array."""

        self._data[i], self._data[j] = self._data[j], self._data[i]



    def _upheap(self, j):

        parent = self._parent(j)

        if j > 0 and self._data[j] < self._data[parent]:

            self._swap(j, parent)

            self._upheap(parent)  # recur at position of parent



    def _downheap(self, j):

        if self._has_left(j):

            left = self._left(j)

            small_child = left  # although right may be smaller

            if self._has_right(j):

                right = self._right(j)

                if self._data[right] < self._data[left]:

                    small_child = right

            if self._data[small_child] < self._data[j]:

                self._swap(j, small_child)

                self._downheap(small_child)  # recur at position of small child



    # ------------------------------ public behaviors ------------------------------

    def __init__(self):

        """Create a new empty Priority Queue."""

        self._data = []



    def __len__(self):

        """Return the number of items in the priority queue."""

        return len(self._data)



    def add(self, key, value):

        """Add a key-value pair to the priority queue."""

        self._data.append(self._Item(key, value))

        self._upheap(len(self._data) - 1)  # upheap newly added position



    def min(self):

        """Return but do not remove (k,v) tuple with minimum key.



        Raise Empty exception if empty.

        """

        if self.is_empty():

            raise Empty('Priority queue is empty.')

        item = self._data[0]

        return (item._key, item._value)



    def remove_min(self):

        """Remove and return (k,v) tuple with minimum key.



        Raise Empty exception if empty.

        """

        if self.is_empty():

            raise Empty('Priority queue is empty.')

        self._swap(0, len(self._data) - 1)  # put minimum item at the end

        item = self._data.pop()  # and remove it from the list;

        self._downheap(0)  # then fix new root

        return (item._key, item._value)
class AdaptableHeapPriorityQueue(HeapPriorityQueue):

    """A locator-based priority queue implemented with a binary heap."""



    # ------------------------------ nested Locator class ------------------------------

    class Locator(HeapPriorityQueue._Item):

        """Token for locating an entry of the priority queue."""

        __slots__ = '_index'  # add index as additional field



        def __init__(self, k, v, j):

            super().__init__(k, v)

            self._index = j



    # ------------------------------ nonpublic behaviors ------------------------------

    # override swap to record new indices

    def _swap(self, i, j):

        super()._swap(i, j)  # perform the swap

        self._data[i]._index = i  # reset locator index (post-swap)

        self._data[j]._index = j  # reset locator index (post-swap)



    def _bubble(self, j):

        if j > 0 and self._data[j] < self._data[self._parent(j)]:

            self._upheap(j)

        else:

            self._downheap(j)



    # ------------------------------ public behaviors ------------------------------

    def add(self, key, value):

        """Add a key-value pair."""

        token = self.Locator(key, value, len(self._data))  # initiaize locator index

        self._data.append(token)

        self._upheap(len(self._data) - 1)

        return token



    def update(self, loc, newkey, newval):

        """Update the key and value for the entry identified by Locator loc."""

        j = loc._index

        if not (0 <= j < len(self) and self._data[j] is loc):

            raise ValueError('Invalid locator')

        loc._key = newkey

        loc._value = newval

        self._bubble(j)



    def remove(self, loc):

        """Remove and return the (k,v) pair identified by Locator loc."""

        j = loc._index

        if not (0 <= j < len(self) and self._data[j] is loc):

            raise ValueError('Invalid locator')

        if j == len(self) - 1:  # item at last position

            self._data.pop()  # just remove it

        else:

            self._swap(j, len(self) - 1)  # swap item to the last position

            self._data.pop()  # remove it from the list

            self._bubble(j)  # fix item displaced by the swap

        return (loc._key, loc._value)
heap = HeapPriorityQueue()

print(heap._data)



heap.add(5, 0)

print(heap._data)



heap.add(1, 0)

print(heap._data)



heap.add(4, 0)

print(heap._data)



heap.add(7, 0)

print(heap._data)



heap.add(3, 0)

print(heap._data)



heap.add(9, 0)

print(heap._data)



heap.add(0, 0)

print(heap._data)



heap.add(2, 0)

print(heap._data)



heap.add(8, 0)

print(heap._data)
heap = AdaptableHeapPriorityQueue()



# Create Initial Heap

heap.add(2, 0)

print(heap._data)



heap.add(3, 0)

print(heap._data)



heap.add(4, 0)

print(heap._data)



heap.add(8, 0)

print(heap._data)



heap.add(5, 0)

print(heap._data)



heap.add(7, 0)

print(heap._data)



heap.add(6, 0)

print(heap._data, '\n')



# Remove 2

minimum = heap.remove(heap._data[0])

print('Removed:', minimum, '\n')



print(heap._data)