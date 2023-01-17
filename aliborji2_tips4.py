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
import numpy as np
u = 3
np.reshape((range(u**2)), [u,u]).tolist()
pow(2.71, 3.14)
2.71 ** 3.14
float('3.14')
str(3.14)
float('inf')
import math
math.isclose(2.13, 2.14, rel_tol = 1e-2)
import random
random.randrange(28)
random.randint(8,16)
random.random()
A = [1, 2, 3, 4, 5, 6]
random.shuffle(A)
print(A)
random.choice(A)
x = [3, 5, 7, 11]
x
y = [1] + [0] * 10
y
list(range(10))
A
A.append(42)
A.remove(2)
A.insert(3, 28)
A

import bisect
# bisect?
def grade(score, breakpoints=[60, 70, 80, 90], grades='FDCBA'):
        i = bisect.bisect(breakpoints, score)
        return grades[i]

[grade(score) for score in [33, 99, 77, 70, 89, 90, 100]]
# ['F', 'A', 'C', 'C', 'B', 'A', 'A']
def is_palindromic(s):
# Note that s[~i] for i in [0, len(s) - 1] is s[-(i + 1)].
    return all(s[i] == s[~i] for i in range(len(s) // 2))
ss = 'teststring'
print(ss[2], '-', ss[~2])    # gives the exact opposite index
print(ss[-2], '-', ss[~-2])
is_palindromic('kerek')
'Euclid,Axiom 5,Parallel Lines'.split(',')
a  = 'xx', 'yy'
a
type(a)
3 * '01', ','.join(('Gauss', 'Prince of Mathematicians', '1777-1855'))
'Name {name}, Rank {rank}'.format(name='Archimedes', rank=3)
'Name {}, Rank {}'.format('Archimedes', 3)
# both work
# s.append(e) pushes an element onto the stack.
# s[-1] will retrieve, but does not remove, the element at the top of the stack.
# s.pop() will remove and return the element at the top of the stack.
# len(s) == 0 tests if the stack is empty
# Some of the problems require you to implement your own queue class; for others, use the collections.deque class.
# q.append(e) pushes an element onto the queue.
# q[0] will retrieve, but not remove, the element at the front of the queue.
# q.popleft() will remove and return the element at the front of the queue.
# Heap functionality in Python is provided by the heapq module.
# heapq.heapify(L), which transforms the elements in L into a heap in-place,
# heapq.nlargest(k, L) (heapq.nsmallest(k, L)) returns the k largest (smallest) elements in L,
# heapq.heappush(h, e), which pushes a new element on the heap,
# heapq.heappop(h), which pops the smallest element from the heap,
# heapq.heappushpop(h, a), which pushes a on the heap and then pops and returns the smallest element, 
# e = h[0], which returns the smallest element on the heap without popping it.
# To find the first element that is not less than a targeted value, use bisect.bisect_left(a,x).
# This call returns the index of the first entry that is greater than or equal to the targeted value.
# If all elements in the list are less than x, the returned value is len(a).

# To find the first element that is greater than a targeted value, use bisect.bisect_right(a,x).
# This call returns the index of the first entry that is greater than the targeted value. If all elements
# in the list are less than or equal to x, the returned value is len(a).
# There are multiple hash table-based data structures commonly used in Python—set, dict,
# collections.defaultdict, and collections.Counter. The difference between set and the other
# three is that is set simply stores keys, whereas the others store key-value pairs. All have the
# property that they do not allow for duplicate keys, unlike, for example, list.
import collections
c = collections.Counter(a=3, b=1)
d = collections.Counter(a=1, b=2)
# add two counters together: c[x] + d[x], collections.Counter({'a': 4, 'b': 3})
print(c + d)
# subtract (keeping only positive counts), collections.Counter({'a': 2})
print(c - d)
# intersection: min(c[x], d[x]), collections.Counter({'a': 1, 'b': 1})
print(c & d)
# union: max(c[x], d[x]), collections.Counter({'a': 3, 'b': 2})
print(c | d)
pp = {1: '3', 2: '3', 20: '3', 10: '3'}
collections.Counter(pp.items())
# The sort() method implements a stable in-place sort for list objects.
# It takes two arguments, both optional: key=None, and reverse=False

a=[1, 2, 4, 3, 5, 0, 11, 21, 100]  
a.sort(key=lambda x: str(x))
a
# For example, 
b = sorted(a,key=lambda x: str(x)) #leaves array a unchanged
def intersect_two_sorted_arrays(A, B):
    return [a for i, a in enumerate(A) if (i == 0 or a != A[i - 1]) and a in B]
n =  [3, 3, 5, 5, 6, 7, 7, 8, 12]
m =  [5, 5, 6, 8, 8, 9, 10, 10]
%timeit x = intersect_two_sorted_arrays(n, m)
print(x)

def intersect_two_sorted_arrays(A, B):
    def is_present(k):
        i = bisect.bisect_left(B, k)
        return i < len(B) and B[i] == k
    return [a for i, a in enumerate(A)
    if (i == 0 or a != A[i - 1]) and is_present(a)]

%timeit x = intersect_two_sorted_arrays(n, m)
print(x)
# # The sortedcontainers module the best-in-class module for sorted sets and sorted dictionaries—
# # it is performant, has a clean API that is well documented, with a responsive community


# In the interests of pedagogy, we have elected to use the bintrees module which implements
# sorted sets and sorted dictionaries using balanced BSTs. However, any reasonable interviewer
# should accept sortedcontainers wherever we use bintrees.
# Below, we describe the functionalities added by bintrees.
#  insert(e) inserts new element e in the BST.
#  discard(e) removes e in the BST if present.
#  min_item()/max_item() yield the smallest and largest key-value pair in the BST.
#  min_key()/max_key() yield the smallest and largest key in the BST.
#  pop_min()/pop_max() remove the return the smallest and largest key-value pair in the BST.


class BSTNode:
    def __init__(self , data=None , left=None , right=None):
        self.data , self.left , self.right = data , left , right
def search_bst(tree , key):
    return (tree if not tree or tree.data == key else search_bst(tree.left , key)
        if key < tree.data else search_bst(tree.right , key))
t = bintrees.RBTree([(5, 'Alfa'), (2, 'Bravo'), (7, 'Charlie'), (3, 'Delta'), (6, 'Echo')])
import sortedcontainers
def gcd(x, y):
    return x if y == 0 else gcd(y, x % y)
gcd(10, 8)
def fibonacci(n, cache={}):
    if n <= 1:
        return n
    elif n not in cache:
        cache[n] = fibonacci(n - 1) + fibonacci(n - 2)
    return cache[n]
%timeit fibonacci(10)
def change_making(cents):
    COINS = [100, 50, 25, 10, 5, 1]
    num_coins = 0
    for coin in COINS:
        num_coins += cents // coin
        cents %= coin
    return num_coins
change_making(420)
print(5/2, 5//2)
a = [2, 1, 2, 4, 7, 11]
# k = 4
def has_two_sum(A, t):
#     return any([i for i in A if t - i in A])
    return any([t - i in A for i in A])

has_two_sum(a, 22)
# or 
# len([i for i in a if k - i in a]) > 1
def has_three_sum(A, t):
    A.sort()
    # Finds if the sum of two numbers in A equals to t - a.
    return [has_two_sum(A, t-a) for a in A]
has_two_sum(a, 23)
increment_by_i = [lambda x: x + i for i in range(10)]
print(increment_by_i[3](4))
def create_increment_function(x):
    return lambda y: y + x
increment_by_i = [create_increment_function(i) for i in range(10)]
print(increment_by_i[3](4))
A = [[1, 2, 3],[4, 5, 6]]
B = A[:]
B[0][0] = -1
print(A)
print(B)
import copy
A = [[1, 2, 3],[4, 5, 6]]
B = copy.copy(A)
B[0][0] = -1

print(A)
print(B)
import copy
A = [[1, 2, 3],[4, 5, 6]]
B = copy.deepcopy(A)
B[0][0] = -1

print(A)
print(B)
A = [4, 1, 5]
B = A[:]
B[0] = -1

print(A)
print(B)
from abc import ABC, abstractmethod
class Room(ABC):
    @abstractmethod
    def connect(self , room2):
        pass
    
class MazeGame(ABC):
    @abstractmethod
    def make_room(self):
        print("abstract make_room")
        pass

    def addRoom(self , room):
        print("adding room")

    def __init__(self):
        room1 = self.make_room()
        room2 = self.make_room()
        room1.connect(room2)
        self.addRoom(room1)
        self.addRoom(room2)    

class MagicMazeGame(MazeGame):
    def make_room(self):
        return MagicRoom()

class MagicRoom(Room):
    def connect(self , room2):
        print("Connecting magic room")
# ordinary_maze_game = ordinary_maze_game.OrdinaryMazeGame()
ordinary_maze_game = MagicMazeGame()
ordinary_maze_game.make_room().connect(10)
