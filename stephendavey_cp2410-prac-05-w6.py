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
class LinkedQueue:

  """FIFO queue implementation using a singly linked list for storage."""



  #-------------------------- nested _Node class --------------------------

  class _Node:

    """Lightweight, nonpublic class for storing a singly linked node."""

    __slots__ = '_element', '_next'         # streamline memory usage



    def __init__(self, element, next):

      self._element = element

      self._next = next



  #------------------------------- queue methods -------------------------------

  def __init__(self):

    """Create an empty queue."""

    self._head = None

    self._tail = None

    self._size = 0                          # number of queue elements



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

    if self.is_empty():

      raise Exception('Queue is empty')

    return self._head._element              # front aligned with head of list



  def dequeue(self):

    """Remove and return the first element of the queue (i.e., FIFO).



    Raise Empty exception if the queue is empty.

    """

    if self.is_empty():

      raise Exception('Queue is empty')

    answer = self._head._element

    self._head = self._head._next

    self._size -= 1

    if self.is_empty():                     # special case as queue is empty

      self._tail = None                     # removed head had been the tail

    return answer



  def enqueue(self, e):

    """Add an element to the back of queue."""

    newest = self._Node(e, None)            # node will be new tail node

    if self.is_empty():

      self._head = newest                   # special case: previously empty

    else:

      self._tail._next = newest

    self._tail = newest                     # update reference to tail node

    self._size += 1





class CircularQueue:

  """Queue implementation using circularly linked list for storage."""



  #---------------------------------------------------------------------------------

  # nested _Node class

  class _Node:

    """Lightweight, nonpublic class for storing a singly linked node."""

    __slots__ = '_element', '_next'         # streamline memory usage



    def __init__(self, element, next):

      self._element = element

      self._next = next



  # end of _Node class

  #---------------------------------------------------------------------------------



  def __init__(self):

    """Create an empty queue."""

    self._tail = None                     # will represent tail of queue

    self._size = 0                        # number of queue elements



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

    if self.is_empty():

      raise Exception('Queue is empty')

    head = self._tail._next

    return head._element



  def dequeue(self):

    """Remove and return the first element of the queue (i.e., FIFO).



    Raise Empty exception if the queue is empty.

    """

    if self.is_empty():

      raise Exception('Queue is empty')

    oldhead = self._tail._next

    if self._size == 1:                   # removing only element

      self._tail = None                   # queue becomes empty

    else:

      self._tail._next = oldhead._next    # bypass the old head

    self._size -= 1

    return oldhead._element



  def enqueue(self, e):

    """Add an element to the back of queue."""

    newest = self._Node(e, None)          # node will be new tail node

    if self.is_empty():

      newest._next = newest               # initialize circularly

    else:

      newest._next = self._tail._next     # new node points to head

      self._tail._next = newest           # old tail points to new node

    self._tail = newest                   # new node becomes the tail

    self._size += 1



  def rotate(self):

    """Rotate front element to the back of the queue."""

    if self._size > 0:

      self._tail = self._tail._next       # old head becomes new tail



class _DoublyLinkedBase:

  """A base class providing a doubly linked list representation."""



  #-------------------------- nested _Node class --------------------------

  # nested _Node class

  class _Node:

    """Lightweight, nonpublic class for storing a doubly linked node."""

    __slots__ = '_element', '_prev', '_next'            # streamline memory



    def __init__(self, element, prev, next):            # initialize node's fields

      self._element = element                           # user's element

      self._prev = prev                                 # previous node reference

      self._next = next                                 # next node reference



  #-------------------------- list constructor --------------------------



  def __init__(self):

    """Create an empty list."""

    self._header = self._Node(None, None, None)

    self._trailer = self._Node(None, None, None)

    self._header._next = self._trailer                  # trailer is after header

    self._trailer._prev = self._header                  # header is before trailer

    self._size = 0                                      # number of elements



  #-------------------------- public accessors --------------------------



  def __len__(self):

    """Return the number of elements in the list."""

    return self._size



  def is_empty(self):

    """Return True if list is empty."""

    return self._size == 0



  #-------------------------- nonpublic utilities --------------------------



  def _insert_between(self, e, predecessor, successor):

    """Add element e between two existing nodes and return new node."""

    newest = self._Node(e, predecessor, successor)      # linked to neighbors

    predecessor._next = newest

    successor._prev = newest

    self._size += 1

    return newest



  def _delete_node(self, node):

    """Delete nonsentinel node from the list and return its element."""

    predecessor = node._prev

    successor = node._next

    predecessor._next = successor

    successor._prev = predecessor

    self._size -= 1

    element = node._element                             # record deleted element

    node._prev = node._next = node._element = None      # deprecate node

    return element                                      # return deleted element







class PositionalList(_DoublyLinkedBase):

  """A sequential container of elements allowing positional access."""



  #-------------------------- nested Position class --------------------------

  class Position:

    """An abstraction representing the location of a single element.



    Note that two position instaces may represent the same inherent

    location in the list.  Therefore, users should always rely on

    syntax 'p == q' rather than 'p is q' when testing equivalence of

    positions.

    """



    def __init__(self, container, node):

      """Constructor should not be invoked by user."""

      self._container = container

      self._node = node

    

    def element(self):

      """Return the element stored at this Position."""

      return self._node._element

      

    def __eq__(self, other):

      """Return True if other is a Position representing the same location."""

      return type(other) is type(self) and other._node is self._node



    def __ne__(self, other):

      """Return True if other does not represent the same location."""

      return not (self == other)               # opposite of __eq__

    

  #------------------------------- utility methods -------------------------------

  def _validate(self, p):

    """Return position's node, or raise appropriate error if invalid."""

    if not isinstance(p, self.Position):

      raise TypeError('p must be proper Position type')

    if p._container is not self:

      raise ValueError('p does not belong to this container')

    if p._node._next is None:                  # convention for deprecated nodes

      raise ValueError('p is no longer valid')

    return p._node



  def _make_position(self, node):

    """Return Position instance for given node (or None if sentinel)."""

    if node is self._header or node is self._trailer:

      return None                              # boundary violation

    else:

      return self.Position(self, node)         # legitimate position

    

  #------------------------------- accessors -------------------------------

  def first(self):

    """Return the first Position in the list (or None if list is empty)."""

    return self._make_position(self._header._next)



  def last(self):

    """Return the last Position in the list (or None if list is empty)."""

    return self._make_position(self._trailer._prev)



  def before(self, p):

    """Return the Position just before Position p (or None if p is first)."""

    node = self._validate(p)

    return self._make_position(node._prev)



  def after(self, p):

    """Return the Position just after Position p (or None if p is last)."""

    node = self._validate(p)

    return self._make_position(node._next)



  def __iter__(self):

    """Generate a forward iteration of the elements of the list."""

    cursor = self.first()

    while cursor is not None:

      yield cursor.element()

      cursor = self.after(cursor)



  #------------------------------- mutators -------------------------------

  # override inherited version to return Position, rather than Node

  def _insert_between(self, e, predecessor, successor):

    """Add element between existing nodes and return new Position."""

    node = super()._insert_between(e, predecessor, successor)

    return self._make_position(node)



  def add_first(self, e):

    """Insert element e at the front of the list and return new Position."""

    return self._insert_between(e, self._header, self._header._next)



  def add_last(self, e):

    """Insert element e at the back of the list and return new Position."""

    return self._insert_between(e, self._trailer._prev, self._trailer)



  def add_before(self, p, e):

    """Insert element e into list before Position p and return new Position."""

    original = self._validate(p)

    return self._insert_between(e, original._prev, original)



  def add_after(self, p, e):

    """Insert element e into list after Position p and return new Position."""

    original = self._validate(p)

    return self._insert_between(e, original, original._next)



  def delete(self, p):

    """Remove and return the element at Position p."""

    original = self._validate(p)

    return self._delete_node(original)  # inherited method returns element

  

  def replace(self, p, e):

    """Replace the element at Position p with e.



    Return the element formerly at Position p.

    """

    original = self._validate(p)

    old_value = original._element       # temporarily store old element

    original._element = e               # replace with new element

    return old_value                    # return the old element value



  def max(self):

    """Returns maximum element from PosiionalList.

    Assumes list elements are numbers"""

    # Iterate through PL

    maximum = 0

    for item in self:

        if item > maximum:

            maximum = item

    return maximum
def find_next_to_last(linked_q):

    """Finds second to last node in singly linked list."""

    q = LinkedQueue()

    node_a = []

    end = False

    for i in range(len(linked_q)):

        if len(linked_q) <= 1:

            return node_a

        else:

            node_a = linked_q.dequeue()

q = LinkedQueue()

for i in range(1, 11):

    q.enqueue(i)

a = find_next_to_last(q)

print(a)
def count_nodes(circular_q):

    """Counts number of nodes in a cicrularly linked list/queue"""

    first = circular_q.first()

    nxt = []

    count = 0

    while nxt != first:

        circular_q.rotate()

        nxt = circular_q.first()

        count += 1

    return count
c_q = CircularQueue()

for i in range(10):

    c_q.enqueue(i)

count_nodes(c_q)
def list_to_positional_list(l):

    pos_list = PositionalList()

    for item in l:

        pos_list.add_last(item)

    return pos_list
l = [1,2,3,4,5]

p = list_to_positional_list(l)

pos = p.first()

pos.element()
l = [4,3,78,4,99,1,2]

p = list_to_positional_list(l)

p.max()