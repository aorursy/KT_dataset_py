# Question 1

"""

Algorithm find_second_to_last(L)

    if L.head is None or L.head.next is None:    (check if the list has 0 or 1 element)

        return None                              (there is no second last node)

    current = L.head

    next = L.head.next

    while next.next is not None:                 (if next has a next, then current is not second last)

        current = next

        next = next.next

    return current

"""



# Question 2

def count_nodes(L):

    if L.current is None:

        return 0

    count = 1

    original = L.current

    current = original

    while current.next != original:

        count += 1

    return count



# Question 3

"""

Algorithm same_circular_list(x, y):

    current = x

    while current.next is not y: (loop until y is found, indicating that it’s the same list)

        if current.next is x:    (or if x is found again, then they must be different lists)

            return False

        current = current.next

    return True

"""



# Question 4

from potential_list import PositionalList



def list_to_positional_list(list_):

    """Construct a new PositionalList using the contents of list_"""

    pos_list = PositionalList()

    for element in list_:

        pos_list.add_last(element)

    return pos_list



# Question 5

def max(self):

    """Return the maximal element."""

    return max(element for element in self)
