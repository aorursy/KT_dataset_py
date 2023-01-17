import numpy as np
x = np.array([1,2,3,4])
type(x)
np.zeros((3,4))
np.ones((2,4))
np.ones((2,4))*5
n = np.arange(10,50,5)

n
decimal = np.arange(0,2,0.2)

decimal
lin = np.linspace(0,4,12)

lin
rast = np.random.rand(3,4)

rast
n = np.random.randn(2,5)*5

n
np.eye(5)
c= np.random.randint(1,50,15)

c
y = np.arange(24)

y
y.reshape(6,4)
y.reshape(2,3,4)
c.max()
c.min()
c.argmax()
c.argmin()
a = np.arange(50)

a.shape = 2,-1,5

a.shape
a
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
y.ravel()
y[0:2,1]
y[:,3]
for row in y:

    print(row)
for row in y.flat:

    print(row)
for row in np.ndenumerate(y):

    print(row)
a = np.array([2,3])

b = np.array([5,6])
np.hstack((a,b))
np.vstack((a,b))
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
k = np.random.random(10)
k
k.mean()
np.median(k)
np.std(k)
np.var(k)
copied = k.copy()
copied
city = np.array(["ankara","istanbul","bursa","ankara","izmir","bursa","izmir"])

np.unique(city)