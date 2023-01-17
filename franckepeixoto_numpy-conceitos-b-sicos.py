import numpy as np 
np.zeros(10)
np.zeros((5,5))
zero = np.zeros((5,5))

print("shape: ", zero.shape)

print("ndim:", zero.ndim," - // len(zero.shape)")

print("size:", zero.size)
np.zeros((2,3,4))
np.ones((5,2))
np.full((2,5), np.log(8))
np.array([[2,4,6,8], [8,6,4,2]])
np.arange(10, 25)
np.linspace(6, 4/4, 5)
np.random.rand(2,2), np.random.randn(2,2)
%matplotlib inline

import matplotlib.pyplot as plt

plt.hist(np.random.rand(500), density=True, bins=10, histtype="step", color="green", label="rand")

plt.hist(np.random.randn(500), density=True, bins=10, histtype="step", color="purple", label="randn")

plt.axis([-2.5, 2.5, 0, 1.1])

plt.legend(loc = "upper left")

plt.title("Distribuições Aleatórias")

plt.xlabel("Valores")

plt.ylabel("Density")

plt.show()
arr = np.array([[2,2],[20, 10]], dtype=np.int16)

arr.data
matriz = np.arange(25)



print(matriz)

print("Rank:",matriz.ndim)

print("Shape:",matriz.shape)

print()

print()



matriz = matriz.reshape(5,5)



print(matriz)

print("Rank:",matriz.ndim)

print("Shape:",matriz.shape)

print()
matriz.ravel()
valores = np.array([1, 2, 30, 6])

print(valores > [2, 4, 5, 6])

print()

print("equivalente valores < [5, 5, 5, 5]", valores < 5)
v = np.array([[4, 9, 6], [7, 5, 6]])

print(v)

print()

for func in (v.min, v.max, v.sum, v.prod, v.std, v.var,v.mean):

    print(func.__name__, "=", func())

    print()

values = np.array([[2.5, 3.413, 56], [9, 1.1, 7.2]])

print(values)

print()

for func in (np.abs, np.sqrt, np.exp, np.log, np.sign, np.ceil, np.modf, np.isnan, np.cos):

    print("\n", func.__name__)

    print(func(values))
a = np.array([1, -2, 3, 4])

b = np.array([2, 8, -1, 7])



print("a = ",a)

print("b = ",b)

print()

for func in (np.add,np.subtract,np.minimum, np.greater, np.maximum, np.copysign):

    print("\n", func.__name__)

    print(func(a,b))
a = np.array([1, 3, 5, 7, 9, 11, 13])

print("a[5]= ",a[5])

print("a[3:5]= ",a[3:5])

print("a[4:-1]= ",a[4:-1])

print("a[:3]= ",a[:3])

print("a[2::2]= ",a[2::2])

print("a[::-1]= ",a[::-1])

print()
b = np.arange(48).reshape(6,8)

print(b)

print()

print("b[1, 2]=",b[1, 2])

print("b[2, :]=",b[2, :])

print("b[:, 3]=",b[:, 3])

print("b[0, :]=",b[0, :])

print("b[1:2, :]=",b[1:2, :])