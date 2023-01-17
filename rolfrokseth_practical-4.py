stack = []



stack.append(5)

print(stack)

stack.append(3)

print(stack)

stack.pop()

print(stack)

stack.append(2)

print(stack)

stack.append(8)

print(stack)

stack.pop()

print(stack)

stack.pop()

print(stack)

stack.append(9)

print(stack)

stack.append(1)

print(stack)

stack.pop()

print(stack)

stack.append(7)

print(stack)

stack.append(6)

print(stack)

stack.pop()

print(stack)

stack.pop()

print(stack)

stack.append(4)

print(stack)

stack.pop()

print(stack)

stack.pop()

print(stack)
s = []

t = []

s.append(5)

s.append(6)

s.append(8)



def transfer(s,t):

    while len(s) > 0:

        t.append(s.pop())

        print("s",s)

        print("t",t)

transfer(s,t)
from collections import deque



queue = deque([])

queue.append(5)

print(queue)

queue.append(3)

print(queue)

queue.popleft()

print(queue)

queue.append(2)

print(queue)

queue.append(8)

print(queue)

queue.popleft()

print(queue)

queue.popleft()

print(queue)

queue.append(9)

print(queue)

queue.append(1)

print(queue)

queue.popleft()

print(queue)

queue.append(7)

print(queue)

queue.append(6)

print(queue)

queue.popleft()

print(queue)

queue.popleft()

print(queue)

queue.append(4)

print(queue)

queue.popleft()

print(queue)

queue.popleft()

print(queue)

# Q.6

D = [1,2,3,4,5,6,7,8]

Q = []



while len(D) > 0:

    Q.append(D[0])

    print("Q ",Q)

    D.pop(0)

    print("D ",D)
# Q.7



D = [1,2,3,4,5,6,7,8]

S = []



while len(D) > 0:

    S.append(len(D))

    D.pop()

    print("S ", S)

    print("D ", D)
