# Q1. 

# Found class @ https://dbader.org/blog/python-linked-list





class ListNode:

    """

    A node in a singly-linked list.

    """

    def __init__(self, data=None, next=None):

        self.data = data

        self.next = next



    def __repr__(self):

        return repr(self.data)





class SinglyLinkedList:

    def __init__(self):

        """

        Create a new singly-linked list.

        Takes O(1) time.

        """

        self.head = None



    def __repr__(self):

        """

        Return a string representation of the list.

        Takes O(n) time.

        """

        nodes = []

        curr = self.head

        while curr:

            nodes.append(repr(curr))

            curr = curr.next

        return '[' + ', '.join(nodes) + ']'



    def prepend(self, data):

        """

        Insert a new element at the beginning of the list.

        Takes O(1) time.

        """

        self.head = ListNode(data=data, next=self.head)



    def append(self, data):

        """

        Insert a new element at the end of the list.

        Takes O(n) time.

        """

        if not self.head:

            self.head = ListNode(data=data)

            return

        curr = self.head

        while curr.next:

            curr = curr.next

        curr.next = ListNode(data=data)



    def find(self, key):

        """

        Search for the first element with `data` matching

        `key`. Return the element or `None` if not found.

        Takes O(n) time.

        """

        curr = self.head

        while curr and curr.data != key:

            curr = curr.next

        return curr  # Will be None if not found



    def remove(self, key):

        """

        Remove the first occurrence of `key` in the list.

        Takes O(n) time.

        """

        # Find the element and keep a

        # reference to the element preceding it

        curr = self.head

        prev = None

        while curr and curr.data != key:

            prev = curr

            curr = curr.next

        # Unlink it from the list

        if prev is None:

            self.head = curr.next

        elif curr:

            prev.next = curr.next

            curr.next = None



    def reverse(self):

        """

        Reverse the list in-place.

        Takes O(n) time.

        """

        curr = self.head

        prev_node = None

        next_node = None

        while curr:

            next_node = curr.next

            curr.next = prev_node

            prev_node = curr

            curr = next_node

        self.head = prev_node

        

        

        

def get_second_last(list):

    current = list.head

    next_element = current.next

    while(next_element.next != None):

        current = next_element

        next_element = next_element.next

    return current



my_list = SinglyLinkedList()

my_list.append(1)

my_list.append(2)

my_list.append(3)

my_list.append(4)

my_list.append(5)

print(get_second_last(my_list))
# Q2.

# Found class @ https://github.com/vprusso/youtube_tutorials/blob/master/data_structures/linked_list/circular_linked_list/circular_linked_list_insert.py



class Node:

    def __init__(self, data):

        self.data = data 

        self.next = None





class CircularLinkedList:

    def __init__(self):

        self.head = None 



    def prepend(self, data):

        new_node = Node(data)

        cur = self.head 

        new_node.next = self.head



        if not self.head:

            new_node.next = new_node

        else:

            while cur.next != self.head:

                cur = cur.next

            cur.next = new_node

        self.head = new_node



    def append(self, data):

        if not self.head:

            self.head = Node(data)

            self.head.next = self.head

        else:

            new_node = Node(data)

            cur = self.head

            while cur.next != self.head:

                cur = cur.next

            cur.next = new_node

            new_node.next = self.head





def count_elements(list):

    start = list.head

    count = 1

    next_element = start.next 

    while start != next_element:

        next_element = next_element.next 

        count += 1

    return count





            

my_list = CircularLinkedList()

my_list.append(1)

my_list.append(2)

my_list.append(3)

my_list.append(4)

my_list.append(5)

print(count_elements(my_list))
# Q3.



def is_same_list(x,y):

    x_cur = x.head

    y_cur = y.head

    while(x_cur.data == y_cur.data):

        x_cur = x_cur.next

        y_cur = y_cur.next

        if(x_cur == x.head and y_cur == y.head):

            return True

    return False



x = CircularLinkedList()

x.append(1)

x.append(2)

x.append(3)



y = CircularLinkedList()

y.append(1)

y.append(2)

y.append(3)



print(is_same_list(x,y))
# Q4.

# Found class @ textbook



class _DoublyLinkedBase:

    """A base class providing a doubly linked list representation."""



    # -------------------------- nested _Node class --------------------------

    # nested _Node class

    class _Node:

        """Lightweight, nonpublic class for storing a doubly linked node."""

        __slots__ = '_element', '_prev', '_next'  # streamline memory



        def __init__(self, element, prev, next):  # initialize node's fields

            self._element = element  # user's element

            self._prev = prev  # previous node reference

            self._next = next  # next node reference



    # -------------------------- list constructor --------------------------



    def __init__(self):

        """Create an empty list."""

        self._header = self._Node(None, None, None)

        self._trailer = self._Node(None, None, None)

        self._header._next = self._trailer  # trailer is after header

        self._trailer._prev = self._header  # header is before trailer

        self._size = 0  # number of elements



    # -------------------------- public accessors --------------------------



    def __len__(self):

        """Return the number of elements in the list."""

        return self._size



    def is_empty(self):

        """Return True if list is empty."""

        return self._size == 0



    # -------------------------- nonpublic utilities --------------------------



    def _insert_between(self, e, predecessor, successor):

        """Add element e between two existing nodes and return new node."""

        newest = self._Node(e, predecessor, successor)  # linked to neighbors

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

        element = node._element  # record deleted element

        node._prev = node._next = node._element = None  # deprecate node

        return element  # return deleted element



    

class PositionalList(_DoublyLinkedBase):

    """A sequential container of elements allowing positional access."""



    # -------------------------- nested Position class --------------------------

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

            return not (self == other)  # opposite of __eq__



    # ------------------------------- utility methods -------------------------------

    def _validate(self, p):

        """Return position's node, or raise appropriate error if invalid."""

        if not isinstance(p, self.Position):

            raise TypeError('p must be proper Position type')

        if p._container is not self:

            raise ValueError('p does not belong to this container')

        if p._node._next is None:  # convention for deprecated nodes

            raise ValueError('p is no longer valid')

        return p._node



    def _make_position(self, node):

        """Return Position instance for given node (or None if sentinel)."""

        if node is self._header or node is self._trailer:

            return None  # boundary violation

        else:

            return self.Position(self, node)  # legitimate position



    # ------------------------------- accessors -------------------------------

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



    def max(self):

        return max(element for element in self)



    # ------------------------------- mutators -------------------------------

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

        old_value = original._element  # temporarily store old element

        original._element = e  # replace with new element

        return old_value  # return the old element value

    



def list_to_positional_list(list):

    my_list = PositionalList()

    for element in list:

        my_list.add_last(element)

    return my_list

            

my_list = list_to_positional_list([1,2,3,4,5])         

print(my_list)
# Q5.



def max_of_list(list):

    return max(element for element in list)

    

print(max_of_list([1,2,3,4,5]))