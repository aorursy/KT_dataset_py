# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from collections.abc import MutableMapping



class MapBase(MutableMapping):

  """Our own abstract base class that includes a nonpublic _Item class."""



  #------------------------------- nested _Item class -------------------------------

  class _Item:

    """Lightweight composite to store key-value pairs as map items."""

    __slots__ = '_key', '_value'



    def __init__(self, k, v):

      self._key = k

      self._value = v



    def __eq__(self, other):               

      return self._key == other._key   # compare items based on their keys



    def __ne__(self, other):

      return not (self == other)       # opposite of __eq__



    def __lt__(self, other):               

      return self._key < other._key    # compare items based on their keys







class UnsortedTableMap(MapBase):

  """Map implementation using an unordered list."""



  def __init__(self):

    """Create an empty map."""

    self._table = []                              # list of _Item's

  

  def __getitem__(self, k):

    """Return value associated with key k (raise KeyError if not found)."""

    for item in self._table:

      if k == item._key:

        return item._value

    raise KeyError('Key Error: ' + repr(k))



  def __setitem__(self, k, v):

    """Assign value v to key k, overwriting existing value if present."""

    for item in self._table:

      if k == item._key:                          # Found a match:

        item._value = v                           # reassign value

        return                                    # and quit    

    # did not find match for key

    self._table.append(self._Item(k,v))



  def __delitem__(self, k):

    """Remove item associated with key k (raise KeyError if not found)."""

    for j in range(len(self._table)):

      if k == self._table[j]._key:                # Found a match:

        self._table.pop(j)                        # remove item

        return                                    # and quit    

    raise KeyError('Key Error: ' + repr(k))



  def __len__(self):

    """Return number of items in the map."""

    return len(self._table)



  def __iter__(self):                             

    """Generate iteration of the map's keys."""

    for item in self._table:

      yield item._key                             # yield the KEY





    



from random import randrange         # used to pick MAD parameters



class HashMapBase(MapBase):

  """Abstract base class for map using hash-table with MAD compression.



  Keys must be hashable and non-None.

  """



  def __init__(self, cap=11, p=109345121):

    """Create an empty hash-table map.



    cap     initial table size (default 11)

    p       positive prime used for MAD (default 109345121)

    """

    self._table = cap * [ None ]

    self._n = 0                                   # number of entries in the map

    self._prime = p                               # prime for MAD compression

    self._scale = 1 + randrange(p-1)              # scale from 1 to p-1 for MAD

    self._shift = randrange(p)                    # shift from 0 to p-1 for MAD

    self._k = None



  def _hash_function(self, k):

    self._k = k  # Update 

    return (hash(k)*self._scale + self._shift) % self._prime % len(self._table)



  def __len__(self):

    return self._n



  def __getitem__(self, k):

    j = self._hash_function(k)

    return self._bucket_getitem(j, k)             # may raise KeyError



  def __setitem__(self, k, v):

    j = self._hash_function(k)

    self._bucket_setitem(j, k, v)                 # subroutine maintains self._n

    if self._n > len(self._table) // 2:           # keep load factor <= 0.5

      self._resize(2 * len(self._table) - 1)      # number 2^x - 1 is often prime



  def __delitem__(self, k):

    j = self._hash_function(k)

    self._bucket_delitem(j, k)                    # may raise KeyError

    self._n -= 1



  def _resize(self, c):

    """Resize bucket array to capacity c and rehash all items."""

    old = list(self.items())       # use iteration to record existing items

    self._table = c * [None]       # then reset table to desired capacity

    self._n = 0                    # n recomputed during subsequent adds

    for (k,v) in old:

      self[k] = v                  # reinsert old key-value pair

    

    

    

    

class ChainHashMap(HashMapBase):

  """Hash map implemented with separate chaining for collision resolution."""



  def _bucket_getitem(self, j, k):

    bucket = self._table[j]

    if bucket is None:

      raise KeyError('Key Error: ' + repr(k))        # no match found

    return bucket[k]                                 # may raise KeyError



  def _bucket_setitem(self, j, k, v):

    if self._table[j] is None:

      self._table[j] = UnsortedTableMap()     # bucket is new to the table

    oldsize = len(self._table[j])

    self._table[j][k] = v

    if len(self._table[j]) > oldsize:         # key was new to the table

      self._n += 1                            # increase overall map size



  def _bucket_delitem(self, j, k):

    bucket = self._table[j]

    if bucket is None:

      raise KeyError('Key Error: ' + repr(k))        # no match found

    del bucket[k]                                    # may raise KeyError



  def __iter__(self):

    for bucket in self._table:

      if bucket is not None:                         # a nonempty slot

        for key in bucket:

          yield key





        

class ProbeHashMap(HashMapBase):

  """Hash map implemented with linear probing for collision resolution."""

  _AVAIL = object()       # sentinal marks locations of previous deletions



  def _is_available(self, j):

    """Return True if index j is available in table."""

    return self._table[j] is None or self._table[j] is ProbeHashMap._AVAIL



  def _find_slot(self, j, k):

    """Search for key k in bucket at index j.



    Return (success, index) tuple, described as follows:

    If match was found, success is True and index denotes its location.

    If no match found, success is False and index denotes first available slot.

    """

    firstAvail = None

    while True:                               

      if self._is_available(j):

        if firstAvail is None:

          firstAvail = j                      # mark this as first avail

        if self._table[j] is None:

          return (False, firstAvail)          # search has failed

      elif k == self._table[j]._key:

        return (True, j)                      # found a match

      j = (j + 1) % len(self._table)          # keep looking (cyclically)



  def _bucket_getitem(self, j, k):

    found, s = self._find_slot(j, k)

    if not found:

      raise KeyError('Key Error: ' + repr(k))        # no match found

    return self._table[s]._value



  def _bucket_setitem(self, j, k, v):

    found, s = self._find_slot(j, k)

    if not found:

      self._table[s] = self._Item(k,v)               # insert new item

      self._n += 1                                   # size has increased

    else:

      self._table[s]._value = v                      # overwrite existing



  def _bucket_delitem(self, j, k):

    found, s = self._find_slot(j, k)

    if not found:

      raise KeyError('Key Error: ' + repr(k))        # no match found

    self._table[s] = ProbeHashMap._AVAIL             # mark as vacated



  def __iter__(self):

    for j in range(len(self._table)):                # scan entire table

      if not self._is_available(j):

        yield self._table[j]._key

keys = [12, 44, 13, 88, 23, 94, 11, 39, 20, 16, 5]



def hash_function(i):

    h = (3*i + 5) % 11

    return h



l = [None]*11

for k in keys:

    i = hash_function(k)

    if l[i] == None:  # Set empty

        l[i] = [k]

    else:

        l[i] = l[i] + [k]

print(l)

keys = [12, 44, 13, 88, 23, 94, 11, 39, 20, 16, 5]



probe_map = ProbeHashMap()

d = {k:0 for k in keys}



for k, v in d.items():

    probe_map._bucket_setitem(hash_function(k), k, v)

    

for i in probe_map:

    print(i, end=", ")

    
class QuadraticProbeHashMap(ProbeHashMap):

    

    def _find_slot(self, j, k):

        """Search for key k in bucket at index j.



        Return (success, index) tuple, described as follows:

        If match was found, success is True and index denotes its location.

        If no match found, success is False and index denotes first available slot.

        """

        firstAvail = None

        i = 0

        while True:

            if self._is_available(j):

                if firstAvail is None:

                    firstAvail = j                      # mark this as first avail

                if self._table[j] is None:

                    return (False, firstAvail)          # search has failed

            elif k == self._table[j]._key:

                return (True, j)                      # found a match

            j = (j + i**2) % len(self._table)         # keep looking (quadratically)

            i += 1

            if i == len(self._table):

                print(f"Unable to place key {k} via quadratic probing")

                return (False, None)



            

q = QuadraticProbeHashMap()

keys = [12, 44, 13, 88, 23, 94, 11, 39, 20, 16, 5]

d = {k:0 for k in keys}



for k, v in d.items():

    try:

        q._bucket_setitem(hash_function(k), k, v)

    except:

        print()

    

for i in q:

    print(i, end=", ")
keys = [12, 44, 13, 88, 23, 94, 11, 39, 20, 16, 5]



def second_hash(k):

    return 7 - (k % 7)



l = [None]*11



for k in keys:

    i = hash_function(k)

    if l[i] == None:  # Set empty

        l[i] = [k]

    else:

        for j in range(len(l)-1):

            i = (i + j*second_hash(k)) % len(l)

            if l[i] == None:

                l[i] = [k]

                break

print(l)
l = [[], [],[54, 28, 41], [], [], [18], [], [], [], [], [10,36],[25,38,12,90]]

print(l)

flat_list = [item for elem in l for item in elem]



def hash_func(k):

    return 3*k % 17

table = ['']*19



for k in flat_list:

    table[hash_func(k)] = k



table
example = [

    [1,1,1,0],

    [0,0,0,0],

    [1,1,0,0],

    [1,1,1,1]

]

# Effectivly each row is sorted in descending order. 

def binary_search(data, target, low, high):

    "Altered binary_search returns index of target is found in Python list"

    if low > high: # base case captures all zeros, and all ones?

        return low  # if all 1's low has iterated to len(row) else is 0

    else:

        mid = (low + high) // 2

        if target == data[mid] and data[mid-1] == 1:  # Found First Zero

            return mid

        elif target < data[mid]:  # Is a 1 search higher 

            return binary_search(data, target, mid+1, high)

        else:  # Keep searching descending

            return binary_search(data, target, low, mid-1)
total = 0

target, low = 0, 0

for row in example:

    result = binary_search(row, target, low, len(row)-1)

    total += result

    print(result)

print(f"Total 1's: {total}")