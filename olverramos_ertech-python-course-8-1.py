L = []
L.append(10)
L.append(15)
L.append(2)
L.append(3)
print (L)
print(L.pop())
print(L.pop())
print(L.pop())
print(L.pop())
L = []
L.insert(0,24)
L.insert(0,32)
L.insert(0,5)
L.insert(0,8)
print (L)
print(L.pop())
print(L.pop())
print(L.pop())
print(L.pop())
from collections import deque
C = deque()
C.append(24)
C.append(32)
C.append(5)
C.append(8)
print (C)
print(C.popleft())
print(C.popleft())
print(C.popleft())
print(C.popleft())
L = [7, 4, 2, 1, 6]
C2 = deque(L)
P = deque()
P.append(67)
P.append(5)
P.append(19)
P.append(34)
print (P)
print(P.pop())
print(P.pop())
print(P.pop())
print(P.pop())
cola_limitada = deque(maxlen=5)
cola_limitada.append('Enero')
cola_limitada.append('Febrero')
cola_limitada.append('Marzo')
cola_limitada.append('Abril')
cola_limitada.append('Mayo')
print (cola_limitada)
cola_limitada.append('Junio')
print (cola_limitada)
cola_limitada.append('Julio')
print (cola_limitada)
cola_limitada.rotate()
print (cola_limitada)
cola_limitada.appendleft('Febrero')
print (cola_limitada)
cola_limitada.extend(['Julio', 'Agosto', 'Septiembre'])
print (cola_limitada)
cola_limitada.extendleft(['Febrero', 'Marzo', 'Abril'])
print (cola_limitada)