import numpy as np

x = np.array([1,2,3,4])
print(x)
x
type(x)
x.ndim
np.zeros((3,4))
np.ones((2,4))
np.ones((2,4))*5
n = np.arange(10,50,5)
n
decimal = np.arange(0,2,0.2)
decimal
lin = np.linspace(0,20,4)
lin
rast = np.random.rand(3,4)
rast
n = np.random.randn(2,5)
n
n = np.random.randn(2,5)*4
n
np.eye(4)
c= np.random.randint(1,5,5)
c
y = np.arange(24)
y
y.shape
r=y.reshape(6,4)
r
r.ndim
f=y.reshape(2,3,-1)
f
f.ndim
c=np.arange(1,24)
c
c.max()
c.min()
c.argmax()
c.argmin()
a = np.arange(50)
a.reshape(2,-1,5)


a.ndim
a = np.arange(10)**2
a
a[3]
a[3:6]
a[4] = 1000
a
for i in a:
    print(i*2)
y = np.random.random(16)
y

y = y.reshape(4,4)
y
y[0:2,1]
y.ravel()
y[:,3]
for row in y:
    print(row)
for row in y.flat:
    print(row)
    
a = np.array([2,3])
b = np.array([5,6])
a

b
np.hstack((a,b))

y=np.vstack((a,b))
y
y.ndim
a = np.array([[1,2],[3,4]])
b = np.array([[5,6]])
np.concatenate((a,b),axis = 0)
np.concatenate((a,b.T),axis=1)
a = np.array([20,40,60,80])
a
b= np.arange(4)
b
c = a+b
c
a**2
a>50
a += 10
a
x = np.random.random(12)
x
x.max()
x.min()
x.sum()
b = np.arange(15).reshape(3,5)
b
b.sum(axis = 0)
b.sum(axis = 1)
b.max(axis = 1)
b.cumsum(axis=1)
k = np.random.random(11)
k
np.sort(k)
k.mean()
np.median(k)
np.std(k)
np.var(k)
copied = k.copy()
copied
import numpy as np

x = np.arange(4)
print("x =", x)
print("x + 5 =", x + 5)
print("x - 5 =", x - 5)
print("x * 2 =", x * 2)
print("x / 2 =", x / 2)
print("x // 2 =", x // 2)
print("-x = ", -x)
print("x ** 2 = ", x ** 2)
print("x % 2 = ", x % 2)
print(np.add(3,2))

print(np.add(x,2)) #Addition +
print(np.subtract(x,5)) #Subtraction -
print(np.negative(x)) #Unary negation -
print(np.multiply(x,3)) #Multiplication *
print(np.divide(x,2)) #Division /
print(np.floor_divide(x,2)) #Floor division //
print(np.power(x,2)) #Exponentiation **
print(np.mod(x,2)) #Modulus/remainder **

print(np.multiply(x, x))
