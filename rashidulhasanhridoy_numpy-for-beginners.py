import numpy as np
fruits = ['apple', 'mango', 'orange', 'banana']

print(fruits)
print(fruits[1])
print(fruits[-1])
fruits[2] = 'Pine Apple'

print(fruits)
if 'mango' in fruits:

    print('Yes!')

else:

    print('No!')
len(fruits)
numbers = [1,2,3,4,5,6,7,8,9,0]

new_numbers = np.array(numbers)

print(new_numbers)
name = np.array(['Rashidul', 'Hasan', 'Hridoy'])

print(name)
a = [1,2,3]

b = [3,2,1]

c = np.array([a,b])

print(c)
d = np.array([[1,2,3], [4,5,6]])

print(d)
d.shape
e = np.array([[1,2,3],[4,5,6],[7,8,9]])

e.shape
e
np.diag(e)
np.ones((3,4))
np.ones((3,1))
np.ones((5,8))
np.zeros((3,5))
np.zeros((4,2))
np.eye(4)
np.eye(6)
np.arange(1,10)
np.arange(4,14)
np.arange(1,20,2)
np.arange(0,20,2)
np.arange(0,40,5)
np.arange(0,100,3,float)
np.arange(1,20, 0.5, float)
x = np.arange(1,11)

x
x.reshape(2,5)
x.reshape(5,2)
x.reshape(10,1)
y = np.array([[1,2,3,4],[5,6,7,8]])

y
y.reshape(4,2)
z = np.array([1,2,3,4,5,6,7,8,9,0])

z.reshape(2,5)
np.linspace(0,4,9)
x = np.linspace(0,10,20)

x
np.linspace(10,100,10)
x = np.arange(0,20)

x
x.reshape(5,4)
x
x.resize(5,4)
x
z = np.array([1,2,3,4,5,6,7,8,9])

z.resize(3,3)

z
a = np.array([[1,2,4,5], [4,5,6,3]])

a.resize(3,3)

a
np.array([1, 2, 3] * 3)
np.repeat([1, 2, 3], 3)
x = np.array([1,2,3])

y = np.array([4,5,6])
x+y

# elementwise addition     [1 2 3] + [4 5 6] = [5  7  9]
x-y

# elementwise subtraction  [1 2 3] - [4 5 6] = [-3 -3 -3]
x*y

# elementwise multiplication  [1 2 3] * [4 5 6] = [4  10  18]
x/y

# elementwise divison         [1 2 3] / [4 5 6] = [0.25  0.4  0.5]
x**2

# elementwise power  [1 2 3] ^2 =  [1 4 9]
x.dot(y)

# dot product  1*4 + 2*5 + 3*6
y.dot(x)
a = np.array([x, x**2])

a
a.shape
a.T
a.T.shape
x.dtype
x = x.astype('f')

x.dtype
a = np.array([1,4,5,6,-5,3,7])
a.sum()
a.max()
a.min()
a.mean()
a.std()
a.argmax()
a.argmin()
b = np.arange(14)**2

b
b[0], b[3], b[-1], b[-5]
a[1:7]
a[3:4]
a[-2:]
a[-5:]
a[-1::-3]
a[-1::-5]
r = np.arange(36)

r.resize((6, 6))

r
r[2,2]
r[5,5]
r[5,2]
r[3, 3:6]
r[:2, :-1]
r[-1, ::2]
r[r > 30]
r[r > 30] = 30

r
r2 = r[:3,:3]

r2
r2[:] = 0

r2
r
r_copy = r.copy()

r_copy
r_copy[:] = 10

print(r_copy, '\n')

print(r)
t = np.random.randint(0, 10, (4,3))

t
for row in t:

    print(row)
for i in range(len(t)):

    print(t[i])
for i, row in enumerate(t):

    print('row', i, 'is', row)
t2 = t**2

t2
for i, j in zip(t, t2):

    print(i,'+',j,'=',i+j)