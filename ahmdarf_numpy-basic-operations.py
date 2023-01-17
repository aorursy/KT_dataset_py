import numpy as np
Z = np.array([1, 2, 4, 5])

Z
Z.dtype
type(Z)
Z.cumsum()
Y = np.array([range(i, i+3) for i in [1, 4, 7]])

Y
np.full((3,5), 101, dtype=int) - 1
print(np.arange(1,10).reshape(3,3)[::-1])

print(np.arange(1,10).reshape(3,3)[::, ::-1])

print(np.arange(1,10).reshape(3,3)[::-1, ::-1])
np.arange(1, 10).reshape(3,3)[0]
K = [0, 0, 0, 0, 0]

L = K[:3]

L[0] = 5

print(K)



K = np.array([0, 0, 0, 0, 0])

L = K[:3]

L[0] = 5

print(K)
K = np.zeros((3,3))

K[1] = 5

K
print(np.zeros((3,3)).reshape(1, -1))

print(np.zeros((3,3)).reshape(-1, 1))
x = np.arange(3)

y = np.arange(3, 6)

z = np.arange(6, 9)

k = np.arange(9, 12)

l = np.arange(12, 15)

np.concatenate((x, y, z, k, l))
Z = np.arange(15)

x, y, z = np.split(Z, [5, 10])

print('x = ' + str(x), 'y = ' + str(y), 'z = ' + str(z), sep='\n')
Z = np.zeros((6,6), dtype=int)

print(np.vsplit(Z, 2), np.vsplit(Z, 3), np.hsplit(Z, 3), sep='\n\n')
x = np.zeros(9)

print(x + 5)

print(np.add(x, 5))

print(-x)

print(np.negative(x))
X = np.arange(5)

Y = np.zeros(5, dtype=int)

np.add(X, 10, out=Y)
X = np.arange(1, 10)

Y = np.arange(1, 10).reshape(3,3)

print(X)

print(np.multiply.reduce(X))

print(np.multiply.accumulate(X), '\n')



print(Y)

print(np.multiply.reduce(Y, axis=1))

print(np.multiply.accumulate(Y, axis=1))

print(np.multiply.accumulate(Y))
Z = np.arange(10)

%timeit sum(Z)

%timeit np.sum(Z)
Z = np.random.randint(10, size=(1,3))

print(Z)



print(np.min(Z)) 

print(Z.min())
np.random.seed(0)

np.random.random((3,3))
Z = np.random.randint(-5, 5, size=(3,3))



print(Z[None][None][np.newaxis], '\n\n')

print(Z[:, None][:, np.newaxis][:, None])
Z = np.arange(10)

print(Z)

print(Z < 5)
X = np.array([1, 2, 3, 4, 5])

Y = np.array([1, 7, 7, 7, 5])

print(X == Y)
Z = np.sort(np.random.randint(10, size=10))

print(Z)

print(Z < 6, '\n')



print(np.count_nonzero(Z < 6))

print(np.sum(Z < 6))

print(np.count_nonzero((Z < 6) == 0))

print(np.sum((Z < 6) == 0))

Z = np.sort(np.random.randint(10, size=(3,3)))

print(Z)

print(Z < 6, '\n')



print(np.sum(Z < 6, axis=1))

print(np.sum(Z < 6, axis=0))
Z = np.sort(np.random.randint(10, size=(3,3)))

print(Z, '\n')



print(np.any(Z > 8, axis=1))

print(np.all(Z > 5, axis=0))
N = np.nan

np.array([1, N, N, N, 5])
Z = np.random.randint(10, size=(3,3))

print(Z)

Z[Z > 5]
print(bool(42))

print(bool(0))

print(bool(-5.5))
A = np.array([1, 0, 1, 0])

B = np.array([1, 0, 0, 1])

print(A & B)

print(A | B)



A = np.array([1, 0, 1, 0], dtype=bool)

B = np.array([1, 0, 0, 1], dtype=bool)

print(A & B)

print(A | B)
Z = np.random.randint(10, size=(3,3))

print(Z)

print((Z > 3) & (Z < 6))

print((Z > 3) | (Z < 6))
Z = np.arange(5)

Z = np.add.outer(Z, Z)

print(Z)

print(Z[3:, 3:])

print(Z[:, [0, 2, 4]])

print(Z[[0, 2, 4], :])

print(Z[[0, 2, 4], 4:])
Z = np.arange(7)

i = np.array([3, 4])

Z[i] = 99

Z
Z = np.zeros(10, dtype=int)

i = [2, 5, 5, 5, 5, 9, 9]

np.add.at(Z, i, 10)

Z
Z = np.array([1, 2, 7, 2, 7, 0, 1, 8, 0, 5])

Y = np.argsort(Z)

print(Y)

print(Z[Y])
Z = np.random.randint(10, size=(3,3))

print(Z)



print(np.sort(Z, axis=0))

print(np.sort(Z, axis=1))
Z = np.arange(1000)

'%d bytes' % (Z.size * Z.itemsize)
X = np.random.randint(2, size=7)

print(X)

print(np.nonzero(X), '\n')



Y = np.random.choice([True, False], size=7)

print(Y)

print(np.nonzero(Y))
Z = np.arange(25).reshape(5,5)

print(Z)

print(Z[1:-1, 1:-1])

print(Z[2:-2, 2:-2])
X = np.arange(5)



print(np.pad(X, 10, 'edge'))

print(np.pad(X, (2,10), 'edge'))

print(np.pad(X, (2,10), 'constant', constant_values=(7)))

print(np.pad(X, (2,10), 'constant', constant_values=(7, np.arange(10))))
Y = np.arange(9).reshape(3,3)



print(np.pad(Y, ((1, 1), (0, 0)), 'edge'))

print(np.pad(Y, ((0, 0) , (1,1)), 'edge'))

print(np.pad(Y, (1, 1), 'edge'))
Z = np.zeros((5,5))

print(np.pad(Z, (1,1), 'constant', constant_values=(1)))

print(np.pad(Z, pad_width=1, mode='constant', constant_values=(1)))
print(0.3 == 0.3 * 1)

print(0.3 == 3 * 0.1)

print(0.1 + 0.2 == 0.3)

print(0.2 + 0.2 == 0.4)

print(0.2 * 0.2 == 0.04)
print(np.diag(np.arange(3)))

print(np.diag(np.arange(3), k = -1))

print(np.diag(100 + np.arange(3)))
Z = np.arange(100, 109).reshape(3,3)

print(Z)

print(Z.item(4))

print(Z.item((1, 1)))
Z = np.arange(10)

print(np.where(Z == 7))

print(np.where(Z > 3))

print(np.where(Z < 5, Z*10, Z*100))
X = np.arange(1, 10).reshape(3, 3)

print(X)

print(np.unravel_index(4 , (3, 3)))

X[np.unravel_index(4 , (3, 3))]
Z = np.array([0, 1, 2, 3])

print(np.tile(Z, 2))

print(np.tile(Z, (2, 1)))

print(np.tile(Z, (2, 2)))

print()

Y = np.array([[1, 2], [3, 4]])

print(np.tile(Y, 2))

print(np.tile(Y, (2, 1)))
Z = np.arange(1, 10)

Z[(Z > 3) & (Z < 8)] *= -1

print(Z)

#Z[Z > 3 & Z < 8] *= -1

#Z[(Z > 3) and (Z < 8)] *= -1
print(np.copysign(5, -1))

print(np.copysign([1, 1, 1], [-3, 2, -4]))

print(np.copysign([-1, -2, -3], 1))
Z = np.array([-1.7, -0.7, -0.2, 0.2, 0.7, 1.7])

print(np.rint(Z))

print(np.round(Z, 0))

print(np.trunc(Z))

print(np.ceil(Z))

print(np.floor(Z))
X = np.array([1, 2, 3, 4, 5, 6])

Y = np.array([7, 8, 3, 4, 5, 9])

np.intersect1d(X, Y)
print(np.datetime64('today', 'D'))

print(np.datetime64('today', 'D') + np.timedelta64(1, 'D'))

print(np.datetime64('today', 'D') + np.timedelta64(7, 'D'))

print(np.arange('2020-01', '2020-02', dtype='datetime64[D]'))
Z = np.array([1, 2, 3])

np.multiply(Z, 10, out=Z)

Z
Z = np.random.uniform(1, 10, size=3)

print(Z)

print(Z - Z % 1)

print(Z // 1)

print(np.floor(Z))

print(np.ceil(Z) - 1)

print(Z.astype(int).astype(float))

print(np.trunc(Z))
Z = np.zeros((5,5))

Z += np.arange(5)

print(Z)



Z = np.zeros((5,5))

Z += np.arange(5)[:, None]

print(Z)
Z = (x*10 for x in range(3))

np.fromiter(Z, int)
Z = np.random.randint(10, size=(3, 3))

print(Z, '\n')

print(np.sort(Z, axis=None))

print(np.sort(Z))

print(np.sort(Z, axis=0))
Z = np.array([3, 3, 3])

print(Z.flags.writeable)



Z = np.array([3, 3, 3])

Z.flags.writeable = False

print(Z.flags.writeable)
Z = np.array([1, 1, 10, 1, 1])

print(np.amax(Z))



Z[np.where(Z == np.amax(Z))] = 20

print(Z)



Z[Z.argmax()] = 30

print(Z)
Z = np.zeros((5,5), [('x',float),('y',float)])

Z['x'], Z['y'] = np.meshgrid(np.linspace(0,1,5),

                             np.linspace(0,1,5))

print(Z)
np.set_printoptions(precision=3)

Z = np.array([1.564867842352352])

print(Z)



np.set_printoptions(threshold = 5)

Z = np.arange(10)

print(Z)