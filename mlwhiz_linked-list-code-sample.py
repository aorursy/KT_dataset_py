class Node:

    def __init__(self,val):

        self.val = val

        self.next = None
head = Node(1)

tmp = head

for i in range(2,10):

    a = Node(i)

    tmp.next = a

    tmp = a
# A function to print a Linked list

def traversal(head):

    while head:

        print(head.val)

        head = head.next
traversal(head)
def insert(head,val):

    new_head = Node(val)

    new_head.next = head

    return new_head
head = insert(head,10)

head = insert(head,11)
traversal(head)
def reverseList(head):

    newhead = None

    while head:

        tmp = head.next

        head.next = newhead

        newhead = head

        head = tmp

    return newhead
head = reverseList(head)
traversal(head)