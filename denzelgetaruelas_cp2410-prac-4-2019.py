# Question 1

class Stack:

     def __init__(self):

         self.items = []

     def isEmpty(self):

         return self.items == []

     def push(self, item):

         self.items.append(item)

     def pop(self):

         return self.items.pop()

     def peek(self):

         return self.items

     def size(self):

         return len(self.items)

        

s = Stack()



s.push(5) 

print(s.peek()) # Output 5

s.push(3)

print(s.peek()) # Output [5,3]

print(s.pop())  # Output 3

print(s.peek()) # Output [5]

s.push(2)

print(s.peek()) # Output: [5, 2]

s.push(8)

print(s.peek()) # Output: [5, 2, 8]

print(s.pop())  # Output: 8

print(s.peek()) # Output: [5, 2]

print(s.pop())  # Output: 2

print(s.peek()) # Output: [5]

s.push(9)

print(s.peek()) # Output: [5, 9]

s.push(1)

print(s.peek()) # Output: [5, 9, 1]

print(s.pop())  # Output: 1

print(s.peek()) # Output: [5, 9]

s.push(7)

print(s.peek()) # Output: [5, 9, 7]

s.push(6)

print(s.peek()) # Output: [5, 9, 7, 6]

print(s.pop())  # Output: 6

print(s.peek()) # Output: [5, 9, 7]

print(s.pop())  # Output: 7

print(s.peek()) # Output: [5, 9]

s.push(4)

print(s.peek()) # Output: [5, 9, 4]

print(s.pop())  # 4

print(s.peek()) # [5, 9]

print(s.pop())  # 9

print(s.peek()) # [5]



# Question 2

"""

If a pop returns an empty error, this does not remove an element. 

7 pops remove an element each.

25 - 7 = 18

"""



# Question 3

def transfer(s, t):

    while not s.is_empty():

        t.push(s.pop())



# Question 4

"""

enqueue(5): 5

enqueue(3): 5, 3

dequeue(): 3 return 5

enqueue(2): 3, 2

enqueue(8): 3, 2, 8

dequeue(): 2, 8 return 3

dequeue(): 8 return 2

enqueue(9): 8, 9

enqueue(1): 8, 9, 1

dequeue(): 9, 1 return 8

enqueue(7): 9, 1, 7

enqueue(6): 9, 1, 7, 6

dequeue(): 1, 7, 6 return 9

dequeue(): 7, 6 return 1

enqueue(4): 7, 6, 4

dequeue(): 6, 4 return 7

dequeue(): 4 return 6

"""



# Question 5

"""

5 Empty errors from dequeues will not have removed an element. So 10 dequeues remove an

element each. 32 - 10 = 22.

"""



# Question 6

"""

Q.enqueue(D.delete_first())

Repeat this 8 times to move all the numbers across

"""



# Question 7

"""

S.push(D.delete_last())

Repeat this 8 times to move all the numbers across

"""