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
import collections.abc

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

      raise Empty('Queue is empty')

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





    

class Tree:

  """Abstract base class representing a tree structure."""



  #------------------------------- nested Position class -------------------------------

  class Position:

    """An abstraction representing the location of a single element within a tree.



    Note that two position instaces may represent the same inherent location in a tree.

    Therefore, users should always rely on syntax 'p == q' rather than 'p is q' when testing

    equivalence of positions.

    """



    def element(self):

      """Return the element stored at this Position."""

      raise NotImplementedError('must be implemented by subclass')

      

    def __eq__(self, other):

      """Return True if other Position represents the same location."""

      raise NotImplementedError('must be implemented by subclass')



    def __ne__(self, other):

      """Return True if other does not represent the same location."""

      return not (self == other)            # opposite of __eq__



  # ---------- abstract methods that concrete subclass must support ----------

  def root(self):

    """Return Position representing the tree's root (or None if empty)."""

    raise NotImplementedError('must be implemented by subclass')



  def parent(self, p):

    """Return Position representing p's parent (or None if p is root)."""

    raise NotImplementedError('must be implemented by subclass')



  def num_children(self, p):

    """Return the number of children that Position p has."""

    raise NotImplementedError('must be implemented by subclass')



  def children(self, p):

    """Generate an iteration of Positions representing p's children."""

    raise NotImplementedError('must be implemented by subclass')



  def __len__(self):

    """Return the total number of elements in the tree."""

    raise NotImplementedError('must be implemented by subclass')



  # ---------- concrete methods implemented in this class ----------

  def is_root(self, p):

    """Return True if Position p represents the root of the tree."""

    return self.root() == p



  def is_leaf(self, p):

    """Return True if Position p does not have any children."""

    return self.num_children(p) == 0



  def is_empty(self):

    """Return True if the tree is empty."""

    return len(self) == 0



  def depth(self, p):

    """Return the number of levels separating Position p from the root."""

    if self.is_root(p):

      return 0

    else:

      return 1 + self.depth(self.parent(p))



  def _height1(self):                 # works, but O(n^2) worst-case time

    """Return the height of the tree."""

    return max(self.depth(p) for p in self.positions() if self.is_leaf(p))



  def _height2(self, p):                  # time is linear in size of subtree

    """Return the height of the subtree rooted at Position p."""

    if self.is_leaf(p):

      return 0

    else:

      return 1 + max(self._height2(c) for c in self.children(p))



  def height(self, p=None):

    """Return the height of the subtree rooted at Position p.



    If p is None, return the height of the entire tree.

    """

    if p is None:

      p = self.root()

    return self._height2(p)        # start _height2 recursion



  def __iter__(self):

    """Generate an iteration of the tree's elements."""

    for p in self.positions():                        # use same order as positions()

      yield p.element()                               # but yield each element



  def positions(self):

    """Generate an iteration of the tree's positions."""

    return self.preorder()                            # return entire preorder iteration



  def preorder(self):

    """Generate a preorder iteration of positions in the tree."""

    if not self.is_empty():

      for p in self._subtree_preorder(self.root()):  # start recursion

        yield p



  def _subtree_preorder(self, p):

    """Generate a preorder iteration of positions in subtree rooted at p."""

    yield p                                           # visit p before its subtrees

    for c in self.children(p):                        # for each child c

      for other in self._subtree_preorder(c):         # do preorder of c's subtree

        yield other                                   # yielding each to our caller



  def postorder(self):

    """Generate a postorder iteration of positions in the tree."""

    if not self.is_empty():

      for p in self._subtree_postorder(self.root()):  # start recursion

        yield p



  def _subtree_postorder(self, p):

    """Generate a postorder iteration of positions in subtree rooted at p."""

    for c in self.children(p):                        # for each child c

      for other in self._subtree_postorder(c):        # do postorder of c's subtree

        yield other                                   # yielding each to our caller

    yield p                                           # visit p after its subtrees



  def breadthfirst(self):

    """Generate a breadth-first iteration of the positions of the tree."""

    if not self.is_empty():

      fringe = LinkedQueue()             # known positions not yet yielded

      fringe.enqueue(self.root())        # starting with the root

      while not fringe.is_empty():

        p = fringe.dequeue()             # remove from front of the queue

        yield p                          # report this position

        for c in self.children(p):

          fringe.enqueue(c)              # add children to back of queue



        

        

class BinaryTree(Tree):

  """Abstract base class representing a binary tree structure."""



  # --------------------- additional abstract methods ---------------------

  def left(self, p):

    """Return a Position representing p's left child.



    Return None if p does not have a left child.

    """

    raise NotImplementedError('must be implemented by subclass')



  def right(self, p):

    """Return a Position representing p's right child.



    Return None if p does not have a right child.

    """

    raise NotImplementedError('must be implemented by subclass')



  # ---------- concrete methods implemented in this class ----------

  def sibling(self, p):

    """Return a Position representing p's sibling (or None if no sibling)."""

    parent = self.parent(p)

    if parent is None:                    # p must be the root

      return None                         # root has no sibling

    else:

      if p == self.left(parent):

        return self.right(parent)         # possibly None

      else:

        return self.left(parent)          # possibly None



  def children(self, p):

    """Generate an iteration of Positions representing p's children."""

    if self.left(p) is not None:

      yield self.left(p)

    if self.right(p) is not None:

      yield self.right(p)



  def inorder(self):

    """Generate an inorder iteration of positions in the tree."""

    if not self.is_empty():

      for p in self._subtree_inorder(self.root()):

        yield p



  def _subtree_inorder(self, p):

    """Generate an inorder iteration of positions in subtree rooted at p."""

    if self.left(p) is not None:          # if left child exists, traverse its subtree

      for other in self._subtree_inorder(self.left(p)):

        yield other

    yield p                               # visit p between its subtrees

    if self.right(p) is not None:         # if right child exists, traverse its subtree

      for other in self._subtree_inorder(self.right(p)):

        yield other



  # override inherited version to make inorder the default

  def positions(self):

    """Generate an iteration of the tree's positions."""

    return self.inorder()                 # make inorder the default











class LinkedBinaryTree(BinaryTree):

  """Linked representation of a binary tree structure."""



  #-------------------------- nested _Node class --------------------------

  class _Node:

    """Lightweight, nonpublic class for storing a node."""

    __slots__ = '_element', '_parent', '_left', '_right' # streamline memory usage



    def __init__(self, element, parent=None, left=None, right=None):

      self._element = element

      self._parent = parent

      self._left = left

      self._right = right



  #-------------------------- nested Position class --------------------------

  class Position(BinaryTree.Position):

    """An abstraction representing the location of a single element."""



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



  #------------------------------- utility methods -------------------------------

  def _validate(self, p):

    """Return associated node, if position is valid."""

    if not isinstance(p, self.Position):

      raise TypeError('p must be proper Position type')

    if p._container is not self:

      raise ValueError('p does not belong to this container')

    if p._node._parent is p._node:      # convention for deprecated nodes

      raise ValueError('p is no longer valid')

    return p._node



  def _make_position(self, node):

    """Return Position instance for given node (or None if no node)."""

    return self.Position(self, node) if node is not None else None



  #-------------------------- binary tree constructor --------------------------

  def __init__(self):

    """Create an initially empty binary tree."""

    self._root = None

    self._size = 0



  #-------------------------- public accessors --------------------------

  def __len__(self):

    """Return the total number of elements in the tree."""

    return self._size

  

  def root(self):

    """Return the root Position of the tree (or None if tree is empty)."""

    return self._make_position(self._root)



  def parent(self, p):

    """Return the Position of p's parent (or None if p is root)."""

    node = self._validate(p)

    return self._make_position(node._parent)



  def left(self, p):

    """Return the Position of p's left child (or None if no left child)."""

    node = self._validate(p)

    return self._make_position(node._left)



  def right(self, p):

    """Return the Position of p's right child (or None if no right child)."""

    node = self._validate(p)

    return self._make_position(node._right)



  def num_children(self, p):

    """Return the number of children of Position p."""

    node = self._validate(p)

    count = 0

    if node._left is not None:     # left child exists

      count += 1

    if node._right is not None:    # right child exists

      count += 1

    return count



  #-------------------------- nonpublic mutators --------------------------

  def _add_root(self, e):

    """Place element e at the root of an empty tree and return new Position.



    Raise ValueError if tree nonempty.

    """

    if self._root is not None:

      raise ValueError('Root exists')

    self._size = 1

    self._root = self._Node(e)

    return self._make_position(self._root)



  def _add_left(self, p, e):

    """Create a new left child for Position p, storing element e.



    Return the Position of new node.

    Raise ValueError if Position p is invalid or p already has a left child.

    """

    node = self._validate(p)

    if node._left is not None:

      raise ValueError('Left child exists')

    self._size += 1

    node._left = self._Node(e, node)                  # node is its parent

    return self._make_position(node._left)



  def _add_right(self, p, e):

    """Create a new right child for Position p, storing element e.



    Return the Position of new node.

    Raise ValueError if Position p is invalid or p already has a right child.

    """

    node = self._validate(p)

    if node._right is not None:

      raise ValueError('Right child exists')

    self._size += 1

    node._right = self._Node(e, node)                 # node is its parent

    return self._make_position(node._right)



  def _replace(self, p, e):

    """Replace the element at position p with e, and return old element."""

    node = self._validate(p)

    old = node._element

    node._element = e

    return old



  def _delete(self, p):

    """Delete the node at Position p, and replace it with its child, if any.



    Return the element that had been stored at Position p.

    Raise ValueError if Position p is invalid or p has two children.

    """

    node = self._validate(p)

    if self.num_children(p) == 2:

      raise ValueError('Position has two children')

    child = node._left if node._left else node._right  # might be None

    if child is not None:

      child._parent = node._parent   # child's grandparent becomes parent

    if node is self._root:

      self._root = child             # child becomes root

    else:

      parent = node._parent

      if node is parent._left:

        parent._left = child

      else:

        parent._right = child

    self._size -= 1

    node._parent = node              # convention for deprecated node

    return node._element

  

  def _attach(self, p, t1, t2):

    """Attach trees t1 and t2, respectively, as the left and right subtrees of the external Position p.



    As a side effect, set t1 and t2 to empty.

    Raise TypeError if trees t1 and t2 do not match type of this tree.

    Raise ValueError if Position p is invalid or not external.

    """

    node = self._validate(p)

    if not self.is_leaf(p):

      raise ValueError('position must be leaf')

    if not type(self) is type(t1) is type(t2):    # all 3 trees must be same type

      raise TypeError('Tree types must match')

    self._size += len(t1) + len(t2)

    if not t1.is_empty():         # attached t1 as left subtree of node

      t1._root._parent = node

      node._left = t1._root

      t1._root = None             # set t1 instance to empty

      t1._size = 0

    if not t2.is_empty():         # attached t2 as right subtree of node

      t2._root._parent = node

      node._right = t2._root

      t2._root = None             # set t2 instance to empty

      t2._size = 0

class TreeMap(LinkedBinaryTree, MapBase):

  """Sorted map implementation using a binary search tree."""



  #---------------------------- override Position class ----------------------------

  class Position(LinkedBinaryTree.Position):

    def key(self):

      """Return key of map's key-value pair."""

      return self.element()._key



    def value(self):

      """Return value of map's key-value pair."""

      return self.element()._value



  #------------------------------- nonpublic utilities -------------------------------

  def _subtree_search(self, p, k):

    """Return Position of p's subtree having key k, or last node searched."""

    if k == p.key():                                   # found match

      return p                                         

    elif k < p.key():                                  # search left subtree

      if self.left(p) is not None:

        return self._subtree_search(self.left(p), k)   

    else:                                              # search right subtree

      if self.right(p) is not None:

        return self._subtree_search(self.right(p), k)

    return p                                           # unsucessful search



  def _subtree_first_position(self, p):

    """Return Position of first item in subtree rooted at p."""

    walk = p

    while self.left(walk) is not None:                 # keep walking left

      walk = self.left(walk)

    return walk



  def _subtree_last_position(self, p):

    """Return Position of last item in subtree rooted at p."""

    walk = p

    while self.right(walk) is not None:                # keep walking right

      walk = self.right(walk)

    return walk

  

  #--------------------- public methods providing "positional" support ---------------------

  def first(self):

    """Return the first Position in the tree (or None if empty)."""

    return self._subtree_first_position(self.root()) if len(self) > 0 else None



  def last(self):

    """Return the last Position in the tree (or None if empty)."""

    return self._subtree_last_position(self.root()) if len(self) > 0 else None



  def before(self, p):

    """Return the Position just before p in the natural order.



    Return None if p is the first position.

    """

    self._validate(p)                            # inherited from LinkedBinaryTree

    if self.left(p):

      return self._subtree_last_position(self.left(p))

    else:

      # walk upward

      walk = p

      above = self.parent(walk)

      while above is not None and walk == self.left(above):

        walk = above

        above = self.parent(walk)

      return above



  def after(self, p):

    """Return the Position just after p in the natural order.



    Return None if p is the last position.

    """

    self._validate(p)                            # inherited from LinkedBinaryTree

    if self.right(p):

      return self._subtree_first_position(self.right(p))

    else:

      walk = p

      above = self.parent(walk)

      while above is not None and walk == self.right(above):

        walk = above

        above = self.parent(walk)

      return above



  def find_position(self, k):

    """Return position with key k, or else neighbor (or None if empty)."""

    if self.is_empty():

      return None

    else:

      p = self._subtree_search(self.root(), k)

      self._rebalance_access(p)                  # hook for balanced tree subclasses

      return p



  def delete(self, p):

    """Remove the item at given Position."""

    self._validate(p)                            # inherited from LinkedBinaryTree

    if self.left(p) and self.right(p):           # p has two children

      replacement = self._subtree_last_position(self.left(p))

      self._replace(p, replacement.element())    # from LinkedBinaryTree

      p =  replacement

    # now p has at most one child

    parent = self.parent(p)

    self._delete(p)                              # inherited from LinkedBinaryTree

    self._rebalance_delete(parent)               # if root deleted, parent is None

      

  #--------------------- public methods for (standard) map interface ---------------------

  def __getitem__(self, k):

    """Return value associated with key k (raise KeyError if not found)."""

    if self.is_empty():

      raise KeyError('Key Error: ' + repr(k))

    else:

      p = self._subtree_search(self.root(), k)

      self._rebalance_access(p)                  # hook for balanced tree subclasses

      if k != p.key():

        raise KeyError('Key Error: ' + repr(k))

      return p.value()



  def __setitem__(self, k, v):

    """Assign value v to key k, overwriting existing value if present."""

    if self.is_empty():

      leaf = self._add_root(self._Item(k,v))     # from LinkedBinaryTree

    else:

      p = self._subtree_search(self.root(), k)

      if p.key() == k:

        p.element()._value = v                   # replace existing item's value

        self._rebalance_access(p)                # hook for balanced tree subclasses

        return

      else:

        item = self._Item(k,v)

        if p.key() < k:

          leaf = self._add_right(p, item)        # inherited from LinkedBinaryTree

        else:

          leaf = self._add_left(p, item)         # inherited from LinkedBinaryTree

    self._rebalance_insert(leaf)                 # hook for balanced tree subclasses



  def __delitem__(self, k):

    """Remove item associated with key k (raise KeyError if not found)."""

    if not self.is_empty():

      p = self._subtree_search(self.root(), k)

      if k == p.key():

        self.delete(p)                           # rely on positional version

        return                                   # successful deletion complete

      self._rebalance_access(p)                  # hook for balanced tree subclasses

    raise KeyError('Key Error: ' + repr(k))



  def __iter__(self):

    """Generate an iteration of all keys in the map in order."""

    p = self.first()

    while p is not None:

      yield p.key()

      p = self.after(p)



  #--------------------- public methods for sorted map interface ---------------------

  def __reversed__(self):

    """Generate an iteration of all keys in the map in reverse order."""

    p = self.last()

    while p is not None:

      yield p.key()

      p = self.before(p)



  def find_min(self):

    """Return (key,value) pair with minimum key (or None if empty)."""

    if self.is_empty():

      return None

    else:

      p = self.first()

      return (p.key(), p.value())



  def find_max(self):

    """Return (key,value) pair with maximum key (or None if empty)."""

    if self.is_empty():

      return None

    else:

      p = self.last()

      return (p.key(), p.value())



  def find_le(self, k):

    """Return (key,value) pair with greatest key less than or equal to k.



    Return None if there does not exist such a key.

    """

    if self.is_empty():

      return None

    else:

      p = self.find_position(k)

      if k < p.key():

        p = self.before(p)

      return (p.key(), p.value()) if p is not None else None



  def find_lt(self, k):

    """Return (key,value) pair with greatest key strictly less than k.



    Return None if there does not exist such a key.

    """

    if self.is_empty():

      return None

    else:

      p = self.find_position(k)

      if not p.key() < k:

        p = self.before(p)

      return (p.key(), p.value()) if p is not None else None



  def find_ge(self, k):

    """Return (key,value) pair with least key greater than or equal to k.



    Return None if there does not exist such a key.

    """

    if self.is_empty():

      return None

    else:

      p = self.find_position(k)                   # may not find exact match

      if p.key() < k:                             # p's key is too small

        p = self.after(p)

      return (p.key(), p.value()) if p is not None else None



  def find_gt(self, k):

    """Return (key,value) pair with least key strictly greater than k.



    Return None if there does not exist such a key.

    """

    if self.is_empty():

      return None

    else:

      p = self.find_position(k)

      if not k < p.key():                   

        p = self.after(p)

      return (p.key(), p.value()) if p is not None else None

  

  def find_range(self, start, stop):

    """Iterate all (key,value) pairs such that start <= key < stop.



    If start is None, iteration begins with minimum key of map.

    If stop is None, iteration continues through the maximum key of map.

    """

    if not self.is_empty():

      if start is None:

        p = self.first()

      else:

        # we initialize p with logic similar to find_ge

        p = self.find_position(start)

        if p.key() < start:

          p = self.after(p)

      while p is not None and (stop is None or p.key() < stop):

        yield (p.key(), p.value())

        p = self.after(p)



  #--------------------- hooks used by subclasses to balance a tree ---------------------

  def _rebalance_insert(self, p):

    """Call to indicate that position p is newly added."""

    pass



  def _rebalance_delete(self, p):

    """Call to indicate that a child of p has been removed."""

    pass



  def _rebalance_access(self, p):

    """Call to indicate that position p was recently accessed."""

    pass



  #--------------------- nonpublic methods to support tree balancing ---------------------



  def _relink(self, parent, child, make_left_child):

    """Relink parent node with child node (we allow child to be None)."""

    if make_left_child:                           # make it a left child

      parent._left = child

    else:                                         # make it a right child

      parent._right = child

    if child is not None:                         # make child point to parent

      child._parent = parent



  def _rotate(self, p):

    """Rotate Position p above its parent.



    Switches between these configurations, depending on whether p==a or p==b.



          b                  a

         / \                /  \

        a  t2             t0   b

       / \                     / \

      t0  t1                  t1  t2



    Caller should ensure that p is not the root.

    """

    """Rotate Position p above its parent."""

    x = p._node

    y = x._parent                                 # we assume this exists

    z = y._parent                                 # grandparent (possibly None)

    if z is None:            

      self._root = x                              # x becomes root

      x._parent = None        

    else:

      self._relink(z, x, y == z._left)            # x becomes a direct child of z

    # now rotate x and y, including transfer of middle subtree

    if x == y._left:

      self._relink(y, x._right, True)             # x._right becomes left child of y

      self._relink(x, y, False)                   # y becomes right child of x

    else:

      self._relink(y, x._left, False)             # x._left becomes right child of y

      self._relink(x, y, True)                    # y becomes left child of x



  def _restructure(self, x):

    """Perform a trinode restructure among Position x, its parent, and its grandparent.



    Return the Position that becomes root of the restructured subtree.



    Assumes the nodes are in one of the following configurations:



        z=a                 z=c           z=a               z=c  

       /  \                /  \          /  \              /  \  

      t0  y=b             y=b  t3       t0   y=c          y=a  t3 

         /  \            /  \               /  \         /  \     

        t1  x=c         x=a  t2            x=b  t3      t0   x=b    

           /  \        /  \               /  \              /  \    

          t2  t3      t0  t1             t1  t2            t1  t2   



    The subtree will be restructured so that the node with key b becomes its root.



              b

            /   \

          a       c

         / \     / \

        t0  t1  t2  t3



    Caller should ensure that x has a grandparent.

    """

    """Perform trinode restructure of Position x with parent/grandparent."""

    y = self.parent(x)

    z = self.parent(y)

    if (x == self.right(y)) == (y == self.right(z)):  # matching alignments

      self._rotate(y)                                 # single rotation (of y)

      return y                                        # y is new subtree root

    else:                                             # opposite alignments

      self._rotate(x)                                 # double rotation (of x)     

      self._rotate(x)

      return x                                        # x is new subtree root
