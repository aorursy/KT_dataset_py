a = 10
print(a)
type(a)

b = 1.25
type(b)

c = a + 3j
type(c)
a = "Hello World"
print(a)
print(a[:4])

a[3:]

b = '!!!!###'

print(a + b)

print(type(a))
a = [1,2,3,4]
a
type(a)
a[:2]
a[:4]
a[-1:]
a[-2:]
a[1:3]
a[2:4]
for i in a:
  print(i)
a = [1,2,3,4]
b = [5,6]
a + b
a[1] = 100
a
a.append([5,6,7,8,99])
a[-1:]
a = [1,20,3,4]
a.insert(4,10)
a.pop(2)
a

a.remove(20)
a.sort()
a

x,y,z,b = 1,2,3,4
y
a = (1,2,3,4)
type(a)
a[2] = 20
a[:2]
for i in a:
  print(i)
a = {'x' : [1,2], 'y' : [2]}
type(a)

a['x']

a.keys()
a.values()

for i in a:
  print(i)
  print(a[i])
  
a['x'] = [1,2,100]
a
import pandas as pd

a = {'a' : [1,2], 'b' : [2,3]}

ab  = pd.DataFrame(a)
print(ab)

ab = pd.DataFrame({'gajajs' : [1,2,3],
                 'b' :  ['x','y','z']})
print(ab)