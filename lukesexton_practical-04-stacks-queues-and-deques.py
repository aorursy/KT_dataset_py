# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import collections

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
class ArrayStack:

    """LIFO Stack implementation using a Python list as underlying storage."""



    def __init__(self):

        """Create an empty stack."""

        self._data = []  # nonpublic list instance



    def __len__(self):

        """Return the number of elements in the stack."""

        return len(self._data)



    def is_empty(self):

        """Return True if the stack is empty."""

        return len(self._data) == 0



    def push(self, e):

        """Add element e to the top of the stack."""

        self._data.append(e)  # new item stored at end of list



    def top(self):

        """Return (but do not remove) the element at the top of the stack.



        Raise Empty exception if the stack is empty.

        """

        # if self.is_empty():

        # raise Empty('Stack is empty')

        return self._data[-1]  # the last item in the list



    def pop(self):

        """Remove and return the element from the top of the stack (i.e., LIFO).



        Raise Empty exception if the stack is empty.

        """

        # if self.is_empty():

        #  raise Empty('Stack is empty')

        return self._data.pop()  # remove last item from list
class ArrayQueue:

    """FIFO queue implementation using a Python list as underlying storage."""

    DEFAULT_CAPACITY = 10  # moderate capacity for all new queues



    def __init__(self):

        """Create an empty queue."""

        self._data = [None] * ArrayQueue.DEFAULT_CAPACITY

        self._size = 0

        self._front = 0



    def __len__(self):

        """Return the number of elements in the queue."""

        return self._size



    def is_empty(self):

        """Return True if the queue is empty."""

        return self._size == 0



    def first(self):

        """Return (but do not remove) the element at the front of the queue.



        Raise Empty exception if the queue is empty.

        """

        # if self.is_empty():

            # raise Empty('Queue is empty')

        return self._data[self._front]



    def dequeue(self):

        """Remove and return the first element of the queue (i.e., FIFO).



        Raise Empty exception if the queue is empty.

        """

        # if self.is_empty():

          #  raise Empty('Queue is empty')

        answer = self._data[self._front]

        self._data[self._front] = None  # help garbage collection

        self._front = (self._front + 1) % len(self._data)

        self._size -= 1

        return answer



    def enqueue(self, e):

        """Add an element to the back of queue."""

        if self._size == len(self._data):

            self._resize(2 * len(self.data))  # double the array size

        avail = (self._front + self._size) % len(self._data)

        self._data[avail] = e

        self._size += 1



    def _resize(self, cap):  # we assume cap >= len(self)

        """Resize to a new list of capacity >= len(self)."""

        old = self._data  # keep track of existing list

        self._data = [None] * cap  # allocate list with new capacity

        walk = self._front

        for k in range(self._size):  # only consider existing elements

            self._data[k] = old[walk]  # intentionally shift indices

            walk = (1 + walk) % len(old)  # use old size as modulus

        self._front = 0  # front has been realigned
stack = ArrayStack()



stack.push(5)

print(stack._data)



stack.push(3)

print(stack._data)



pop = stack.pop()

print('Return:', pop)

print(stack._data)



stack.push(2)

print(stack._data)



stack.push(8)

print(stack._data)



pop = stack.pop()

print('Return:', pop)

print(stack._data)



pop = stack.pop()

print('Return:', pop)

print(stack._data)



stack.push(9)

print(stack._data)



stack.push(1)

print(stack._data)



pop = stack.pop()

print('Return:', pop)

print(stack._data)



stack.push(7)

print(stack._data)



stack.push(6)

print(stack._data)



pop = stack.pop()

print('Return:', pop)

print(stack._data)



pop = stack.pop()

print('Return:', pop)

print(stack._data)



stack.push(4)

print(stack._data)



pop = stack.pop()

print('Return:', pop)

print(stack._data)



pop = stack.pop()

print('Return:', pop)

print(stack._data)
def transfer(original_stack, transferred_stack):

    while not original_stack.is_empty():

        transferred_stack.push(original_stack.pop())



    return transferred_stack
original_stack = ArrayStack()

original_stack.push(1)

original_stack.push(2)

original_stack.push(3)

original_stack.push(4)

original_stack.push(5)

print("Length of Original Stack: ", len(original_stack))

print("Top of Original Stack: ", original_stack.top())



transferred_stack = ArrayStack()

new_stack = transfer(original_stack, transferred_stack)

print("\nLength of Transferred Stack: ", len(new_stack))

print("Top of Transferred Stack: ", new_stack.top(), '\n')
q = ArrayQueue()

print(q._data)



q.enqueue(5)

print(q._data)



q.enqueue(3)

print(q._data)



dequeue = q.dequeue()

print('Return:', dequeue)



q.enqueue(2)

print(q._data)



q.enqueue(8)

print(q._data)



dequeue = q.dequeue()

print('Return:', dequeue)



dequeue = q.dequeue()

print('Return:', dequeue)



q.enqueue(9)

print(q._data)



q.enqueue(1)

print(q._data)



dequeue = q.dequeue()

print('Return:', dequeue)



q.enqueue(7)

print(q._data)



q.enqueue(6)

print(q._data)



dequeue = q.dequeue()

print('Return:', dequeue)



dequeue = q.dequeue()

print('Return:', dequeue)



q.enqueue(4)

print(q._data)



dequeue = q.dequeue()

print('Return:', dequeue)



dequeue = q.dequeue()

print('Return:', dequeue)



print(q._data)
deque = collections.deque([1, 2, 3, 4, 5, 6, 7, 8])

print(deque)

queue = ArrayQueue()

while not len(deque) == 0:

    queue.enqueue(deque.popleft())

print('Queue - Length:  ', len(queue))

print('Queue - First Element: ', queue.first(), '\n')
deque_two = collections.deque([1, 2, 3, 4, 5, 6, 7, 8])

stack = ArrayStack()

print(deque_two)

while not len(deque_two) == 0:

    stack.push(deque_two.popleft())

print('Stack - Length: ', len(stack))

print('Stack - Top Element: ', stack.top())