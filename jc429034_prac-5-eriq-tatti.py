def nodeCounter(x):

    if x.current is None:

        return 0

    count = 1

    original = x.current

    current = original

    while current.next != original:

        count += 1

    return count

from positional_list import PositionalList

def list_to_positional_list(list_):

    pos_list = PositionalList()

    for element in list_:

        pos_list.add_last(element)

    return pos_list

def max(self):

    return max(element for element in self)