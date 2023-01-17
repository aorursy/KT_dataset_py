# Q3



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

         return self.items[len(self.items)-1]

        

def transfer(S, T):

    while not S.isEmpty():

        T.push(S.pop())

        

s = Stack()

s.push(1)

s.push(2)

s.push(3)



t = Stack()



transfer(s,t)
# Q6



import collections

import queue



d = collections.deque([1,2,3,4,5,6,7,8])

q = queue.Queue()



q.put(d.popleft())

q.put(d.popleft())

q.put(d.popleft())

d.append(d.popleft())

q.put(d.popleft())

q.put(d.pop())

q.put(d.popleft())

q.put(d.popleft())

q.put(d.popleft())



while not q.empty():

    d.append(q.get())



print(q)

print(d)
# Q7



import collections

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

         return self.items[len(self.items)-1]



d = collections.deque([1,2,3,4,5,6,7,8])

s = Stack()



s.push(d.pop())

s.push(d.pop())

s.push(d.pop())

d.appendleft(d.pop())

s.push(d.pop())

s.push(d.popleft())

s.push(d.pop())

s.push(d.pop())

s.push(d.pop())



while not s.isEmpty():

    d.append(s.pop())



print(s)

print(d)