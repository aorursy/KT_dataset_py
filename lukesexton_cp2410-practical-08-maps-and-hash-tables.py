# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from collections import MutableMapping

from random import randrange

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
class MapBase(MutableMapping):

    """Our own abstract base class that includes a nonpublic _Item class."""



    # ------------------------------- nested _Item class -------------------------------

    class _Item:

        """Lightweight composite to store key-value pairs as map items."""

        __slots__ = '_key', '_value'



        def __init__(self, k, v):

            self._key = k

            self._value = v



        def __eq__(self, other):

            return self._key == other._key  # compare items based on their keys



        def __ne__(self, other):

            return not (self == other)  # opposite of __eq__



        def __lt__(self, other):

            return self._key < other._key  # compare items based on their keys
class UnsortedTableMap(MapBase):

    """Map implementation using an unordered list."""



    def __init__(self):

        """Create an empty map."""

        self._table = []  # list of _Item's



    def __getitem__(self, k):

        """Return value associated with key k (raise KeyError if not found)."""

        for item in self._table:

            if k == item._key:

                return item._value

        raise KeyError('Key Error: ' + repr(k))



    def __setitem__(self, k, v):

        """Assign value v to key k, overwriting existing value if present."""

        for item in self._table:

            if k == item._key:  # Found a match:

                item._value = v  # reassign value

                return  # and quit

        # did not find match for key

        self._table.append(self._Item(k, v))



    def __delitem__(self, k):

        """Remove item associated with key k (raise KeyError if not found)."""

        for j in range(len(self._table)):

            if k == self._table[j]._key:  # Found a match:

                self._table.pop(j)  # remove item

                return  # and quit

        raise KeyError('Key Error: ' + repr(k))



    def __len__(self):

        """Return number of items in the map."""

        return len(self._table)



    def __iter__(self):

        """Generate iteration of the map's keys."""

        for item in self._table:

            yield item._key  # yield the KEY
class HashMapBase(MapBase):

    """Abstract base class for map using hash-table with MAD compression.



    Keys must be hashable and non-None.

    """



    def __init__(self, cap=11, p=109345121):

        """Create an empty hash-table map.



        cap     initial table size (default 11)

        p       positive prime used for MAD (default 109345121)

        """

        self._table = cap * [None]

        self._n = 0  # number of entries in the map

        self._prime = p  # prime for MAD compression

        self._scale = 1 + randrange(p - 1)  # scale from 1 to p-1 for MAD

        self._shift = randrange(p)  # shift from 0 to p-1 for MAD



    def _hash_function(self, k):

        return (hash(k) * self._scale + self._shift) % self._prime % len(self._table)



    def __len__(self):

        return self._n



    def __getitem__(self, k):

        j = self.hash_function(k)

        return self._bucket_getitem(j, k)  # may raise KeyError



    def __setitem__(self, k, v):

        j = self._hash_function(k)

        self._bucket_setitem(j, k, v)  # subroutine maintains self._n

        if self._n > len(self._table) // 2:  # keep load factor <= 0.5

            self._resize(2 * len(self._table) - 1)  # number 2^x - 1 is often prime



    def __delitem__(self, k):

        j = self._hash_function(k)

        self._bucket_delitem(j, k)  # may raise KeyError

        self._n -= 1



    def _resize(self, c):

        """Resize bucket array to capacity c and rehash all items."""

        old = list(self.items())  # use iteration to record existing items

        self._table = c * [None]  # then reset table to desired capacity

        self._n = 0  # n recomputed during subsequent adds

        for (k, v) in old:

            self[k] = v  # reinsert old key-value pair

    

    def hash_function(self, k):

        calculation = (3*k + 5) % 11

        return calculation

    

    def quadratic_hash_function(self, k, j):

        calculation = (((3*k) + 5) + (j**2)) % 11

        return calculation

    

    def secondary_hash_function(self, k, j):

        calculation = (((3*k) + 5) + (j * (7 - (k % 7)))) % 11

        return calculation

    

    def rehash_function(self, k):

        calculation = (3 * k) % 17

        return calculation
class ChainHashMap(HashMapBase):

    """Hash map implemented with separate chaining for collision resolution."""



    def _bucket_getitem(self, j, k):

        bucket = self._table[j]

        if bucket is None:

            raise KeyError('Key Error: ' + repr(k))  # no match found

        return bucket[k]  # may raise KeyError



    def _bucket_setitem(self, j, k, v):

        if self._table[j] is None:

            self._table[j] = UnsortedTableMap()  # bucket is new to the table

        oldsize = len(self._table[j])

        self._table[j][k] = v

        if len(self._table[j]) > oldsize:  # key was new to the table

            self._n += 1  # increase overall map size



    def _bucket_delitem(self, j, k):

        bucket = self._table[j]

        if bucket is None:

            raise KeyError('Key Error: ' + repr(k))  # no match found

        del bucket[k]  # may raise KeyError



    def __iter__(self):

        for bucket in self._table:

            if bucket is not None:  # a nonempty slot

                for key in bucket:

                    yield key

                    
# Create initial hash table

hash_table = ChainHashMap()

keys = {12: 1, 44: 2, 13: 3, 88: 4, 23: 5, 94: 6, 11: 7, 39: 8, 20: 9, 16: 10, 5: 11}



# Set items to buckets in hash table

for key, value in keys.items():

    buckets = []

    print('Key:', key)

    bucket = hash_table.hash_function(key) # perform hash function on all keys creating bucket destination

    buckets.append(bucket)

    for bucket in buckets: # Set key and values to bucket destination 

        hash_table._bucket_setitem(bucket, key, value)

        print('Bucket:', bucket)

        print('Value:', value, '\n')

    
class ProbeHashMap(HashMapBase):

    """Hash map implemented with linear probing for collision resolution."""

    _AVAIL = object()  # sentinal marks locations of previous deletions



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

                    firstAvail = j  # mark this as first avail

                if self._table[j] is None:

                    return (False, firstAvail)  # search has failed

            elif k == self._table[j]._key:

                return (True, j)  # found a match

            j = (j + 1) % len(self._table)  # keep looking (cyclically)



    def _bucket_getitem(self, j, k):

        found, s = self._find_slot(j, k)

        if not found:

            raise KeyError('Key Error: ' + repr(k))  # no match found

        return self._table[s]._value



    def _bucket_setitem(self, j, k, v):

        found, s = self._find_slot(j, k)

        if not found:

            self._table[s] = self._Item(k, v)  # insert new item

            self._n += 1  # size has increased

        else:

            self._table[s]._value = v  # overwrite existing



    def _bucket_delitem(self, j, k):

        found, s = self._find_slot(j, k)

        if not found:

            raise KeyError('Key Error: ' + repr(k))  # no match found

        self._table[s] = ProbeHashMap._AVAIL  # mark as vacated



    def __iter__(self):

        for j in range(len(self._table)):  # scan entire table

            if not self._is_available(j):

                yield self._table[j]._key
hash_table2 = ProbeHashMap()

keys = {12: 1, 44: 2, 13: 3, 88: 4, 23: 5, 94: 6, 11: 7, 39: 8, 20: 9, 16: 10, 5: 11}



for key, value in keys.items():

    buckets = []

    print('Key:', key, '--', 'Value:', value,)

    bucket = hash_table2.hash_function(key) # perform hash function on all keys creating bucket destination

    buckets.append(bucket)

    for bucket in buckets: # Set key and values to bucket destination

        availability = hash_table2._is_available(bucket)

        print('Is bucket', bucket, 'available?', availability)

        while availability == False:

            bucket += 1

            if bucket > 10:

                bucket = 0

                availability = hash_table2._is_available(bucket)

            availability = hash_table2._is_available(bucket)

            print('Checking bucket..', bucket)

        else:

            hash_table2._bucket_setitem(bucket, key, value)

            print('Key is stored in bucket:', bucket, '\n')
hash_table3 = ProbeHashMap()

keys = {12: 1, 44: 2, 13: 3, 88: 4, 23: 5, 94: 6, 11: 7, 39: 8, 20: 9, 16: 10, 5: 11}

            

for key, value in keys.items():

    quadratic_factor = 0

    buckets = []

    print('Key:', key, '--', 'Value:', value,)

    bucket = hash_table3.quadratic_hash_function(key, quadratic_factor) 

    buckets.append(bucket)

    for bucket in buckets: # Set key and values to bucket destination

        availability = hash_table3._is_available(bucket)

        print('Is bucket', bucket, 'available?', availability)

        while availability == False:

            quadratic_factor += 1

            print('Try again with quadratic factor:', quadratic_factor)

            bucket = hash_table3.quadratic_hash_function(key, quadratic_factor)

            availability = hash_table3._is_available(bucket)

            print('Is bucket', bucket, 'available?', availability)

            

            if quadratic_factor == 10:

                print('Unable to find available bucket to store key..')

                break

        else:

            hash_table3._bucket_setitem(bucket, key, value)

            print('Key is stored in bucket:', bucket, 'using a quadratic factor of', quadratic_factor, '\n')
hash_table4 = ProbeHashMap()

keys = {12: 1, 44: 2, 13: 3, 88: 4, 23: 5, 94: 6, 11: 7, 39: 8, 20: 9, 16: 10, 5: 11}

            

for key, value in keys.items():

    multiplier = 0

    buckets = []

    print('Key:', key, '--', 'Value:', value,)

    bucket = hash_table4.hash_function(key)

    buckets.append(bucket)

    for bucket in buckets: # Set key and values to bucket destination

        availability = hash_table4._is_available(bucket)

        print('Is bucket', bucket, 'available?', availability)

        while availability == False:

            multiplier += 1

            print('Try again with secondary hash function with multiplier:', multiplier)

            bucket = hash_table4.secondary_hash_function(key, multiplier)

            availability = hash_table4._is_available(bucket)

            print('Is bucket', bucket, 'available?', availability)

        else:

            hash_table4._bucket_setitem(bucket, key, value)

            print('Key is stored in bucket:', bucket, '\n')
hash_table5 = ChainHashMap(19)

keys = {54: 1, 28: 2, 41: 3, 18: 4, 10: 5, 36: 6, 25: 7, 38: 8, 12: 9, 90: 10}

            

# Set items to buckets in hash table

for key, value in keys.items():

    buckets = []

    print('Key:', key, '--', 'Value:', value, )

    bucket = hash_table5.rehash_function(key) # perform hash function on all keys creating bucket destination

    buckets.append(bucket)

    for bucket in buckets: # Set key and values to bucket destination 

        hash_table5._bucket_setitem(bucket, key, value)

        print('Bucket:', bucket, '\n')