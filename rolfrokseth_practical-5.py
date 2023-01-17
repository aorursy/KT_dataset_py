# From

# https://www.tutorialspoint.com/python/python_linked_lists.htm

class Node:

    def __init__(self, dataval=None):

        self.dataval = dataval

        self.nextval = None



class SLinkedList:

    def __init__(self):

        self.headval = None



    def listprint(self):

        printval = self.headval

        while printval is not None:

            print (printval.dataval)

            printval = printval.nextval

            



list = SLinkedList()

list.headval = Node("Mon")

e2 = Node("Tue")

e3 = Node("Wed")

e4 = Node("Thu")

e5 = Node("Fri")



# Link first Node to second node

list.headval.nextval = e2



# Link second Node to third node

e2.nextval = e3

e3.nextval = e4

e4.nextval = e5



list.listprint()
def second_to_last(list):

    if list.headval is None or list.headval.nextval is None:

        return None

    current = list.headval

    nextv = list.headval.nextval

    while nextv.nextval is not None:

        current = nextv

        nextv = nextv.nextval

    return current.dataval

    

second_to_last(list)
# from https://www.geeksforgeeks.org/circular-linked-list-set-2-traversal/

# Python program to create a circular linked list

# circular linked list traversal  

import sys  

# Structure for a Node 

class Node: 

      

    # Constructor to create  a new node 

    def __init__(self, data): 

        self.data = data  

        self.next = None

  

class CircularLinkedList: 

      

    # Constructor to create a empty circular linked list 

    def __init__(self): 

        self.head = None

  

    # Function to insert a node at the beginning of a 

    # circular linked list 

    def push(self, data): 

        ptr1 = Node(data) 

        temp = self.head 

          

        ptr1.next = self.head 

  

        # If linked list is not None then set the next of 

        # last node 

        if self.head is not None: 

            while(temp.next != self.head): 

                temp = temp.next 

            temp.next = ptr1 

  

        else: 

            ptr1.next = ptr1 # For the first node 

  

        self.head = ptr1  

  

    # Function to print nodes in a given circular linked list 

    def printList(self): 

        temp = self.head 

        if self.head is not None: 

            while(True): 

                print ("%d" %(temp.data)), 

                temp = temp.next

                if (temp == self.head): 

                    break 

                    

    def count(self):

        temp = self.head

        count = 0

        if self.head is not None:

            while(True):

                count += 1

                temp = temp.next

                if (temp == self.head):

                    print(("There are {} nodes in the circular linked list").format(count))

                    break

                    

                    

    def same_list(self,x, y):

        current = self.head

        while current.next.data != y:

            if current.next.data == x:

                print(("The values {} and {} are not both in the list").format(x,y))

                sys.exit()

            current = current.next

            print(current.data)

        print(("The values {} and {} are both in the checked list").format(x,y))

                  

    

# Initialize list as empty 

cllist = CircularLinkedList() 

  

# Created linked list will be 11->2->56->12 

cllist.push(12) 

cllist.push(56) 

cllist.push(2) 

cllist.push(11) 

cllist.push(5)



#print "Contents of circular Linked List"

cllist.printList()



cllist.count()

print("\nQuestion 3\n")





cl1 = CircularLinkedList() 

cl2 = CircularLinkedList() 



cl1.push(5)

cl1.push(10)

cl1.push(40)

cl1.push(75)

cl2.push(5)

cl2.push(100)

cl2.push(300)

cl2.push(400)

print("List 1")

cl1.printList()

print("\nList 2")

cl2.printList()



print("\nSame list check values checked, list 1 values 75,5")

cl1.same_list(75,5)



#print("\nSame list check values checked, list 2 values 400,200")

#cl2.same_list(400,200)


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



  def max_of_list(list):

    return max(element for element in list)



plist = [1,2,3,4,5]



def list_to_positional_list(self):

    pl1 = PositionalList()

    for element in plist:

        pl1.add_last(element)

        print("Adding",element,"to positional list")

    #print(pl1.max_of_list())

    #return pl1.plmax()

    return pl1.__len__()



list_to_positional_list(plist)

#pl1.__iter__()

#pl1.__len__()

#pl1.max_of_list


plist = [1,2,3,4,5]



def list_to_positional_list(self):

    pl1 = PositionalList()

    for element in plist:

        pl1.add_last(element)

        print("Adding",element,"to positional list")

    #return pl1.element()

    #return pl1.plmax()



list_to_positional_list(plist)

#pl1.__iter__()

#pl1.__len__()

pl1.max_of_list