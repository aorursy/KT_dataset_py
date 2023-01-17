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
"""Basic example of an adapter class to provide a stack interface to Python's list."""



class ArrayStack:

  """LIFO Stack implementation using a Python list as underlying storage."""



  def __init__(self):

    """Create an empty stack."""

    self._data = []                       # nonpublic list instance



  def __len__(self):

    """Return the number of elements in the stack."""

    return len(self._data)



  def is_empty(self):

    """Return True if the stack is empty."""

    return len(self._data) == 0



  def push(self, e):

    """Add element e to the top of the stack."""

    self._data.append(e)                  # new item stored at end of list



  def top(self):

    """Return (but do not remove) the element at the top of the stack.



    Raise Empty exception if the stack is empty.

    """

    if self.is_empty():

      raise Empty('Stack is empty')

    return self._data[-1]                 # the last item in the list



  def pop(self):

    """Remove and return the element from the top of the stack (i.e., LIFO).



    Raise Empty exception if the stack is empty.

    """

    if self.is_empty():

      raise Exception('Stack is empty')

    return self._data.pop()               # remove last item from list



# if __name__ == '__main__':

#   S = ArrayStack()                 # contents: [ ]

#   S.push(5)                        # contents: [5]

#   S.push(3)                        # contents: [5, 3]

#   print(len(S))                    # contents: [5, 3];    outputs 2

#   print(S.pop())                   # contents: [5];       outputs 3

#   print(S.is_empty())              # contents: [5];       outputs False

#   print(S.pop())                   # contents: [ ];       outputs 5

#   print(S.is_empty())              # contents: [ ];       outputs True

#   S.push(7)                        # contents: [7]

#   S.push(9)                        # contents: [7, 9]

#   print(S.top())                   # contents: [7, 9];    outputs 9

#   S.push(4)                        # contents: [7, 9, 4]

#   print(len(S))                    # contents: [7, 9, 4]; outputs 3

#   print(S.pop())                   # contents: [7, 9];    outputs 4

#   S.push(6)                        # contents: [7, 9, 6]

#   S.push(8)                        # contents: [7, 9, 6, 8]

#   print(S.pop())                   # contents: [7, 9, 6]; outputs 8



def transfer(s, t):

    # t is a reversed s

    n = len(s)

    l = []

    for i in range(n):

        l.append(s.pop())

    for i in range(n):

        t.push(l[i])

    return t

        
s = ArrayStack()

t = ArrayStack()

s.push(9)

s.push(4)

s.push(7)

print(s.top())

transfer(s, t)

print(f"S is empty: {s.is_empty()}")



for i in range(len(t)):

    print(t.pop())
# assume D is queue = (1,2,3,4,5,6,7,8)

from collections import deque

D = deque([1,2,3,4,5,6,7,8])

Q = deque()

print(D)





for i in range(len(D)):

    Q.append(D.popleft())

Q
D = deque([1,2,3,4,5,6,7,8])

S = ArrayStack()



for i in range(len(D)):

    S.push(D.pop())

for i in range(len(S)):

    print(S.pop())