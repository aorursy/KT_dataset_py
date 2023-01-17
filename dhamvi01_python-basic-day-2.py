import numpy as np ## First you need to import numpy package

print(np.arange(10) + 1) ## it can generate number sequnace

print(np.arange(100))

print(np.random.randn(10))
a = np.array([1,2,3.3])
print(a.dtype)
b = a.astype(int)
print(b)
print(b.dtype)

a = np.array([1,2,3.3],dtype='int')
print(a.dtype)
print(a)
#a = np.zeros((3,4))
#a.dtype
print(a)
a[1] = 10
print(a)

#print(np.ones((3,3,4)))

print(np.arange(20))

#print(np.arange(1,20,2))

#print(np.linspace(0,2,5))

print(np.full((3,3),'a'))

#print(np.eye(5))

print(np.random.randn(10))

#print(np.empty((4,3)))
a = np.array([[1,2,3], [4,5,6]])

print(a.shape) ## Shape of an array

print(a.ndim) # for dimentions

print(len(a)) ## length of an array

print(a.size)

print(a.dtype)

b = a.astype(float)

print(np.info(np.arange))

print(np.info(np.random))
a = np.array([[1,2,3],[4,5,6]])
print(a + 10)

print(a * a)

print(1/a)

print(a-a)

a[0,1] = 10
print(a)
a[0,0] = a[0,0] + 10
print(a)
print(a**2)

a = np.array([10,20,30])
b = np.array([4,5,2])

print(a + b)
print(a - b)
print(a * b)
print(a / b)

print(np.exp(a))
print(np.sqrt(a))
print(np.log(a))
a = np.array([10,20,30])
b = np.array([10,20,30])

print(a > 15)

print(a == b)
print(a != b)
print(a >= b)
print(a <= b)

print(np.array_equal(a,b))
a = [11,12,1,2,3,4,5,10]
a.sort(reverse=True)
print(a)