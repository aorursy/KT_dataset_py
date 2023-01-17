for i in range(10):

    print(i, end=' ')
for value in [2, 4, 6, 8, 10]:

    # do some operation

    print(value + 1, end=' ')
iter([2, 4, 6, 8, 10])
I = iter([2, 4, 6, 8, 10])
print(next(I))
print(next(I))
print(next(I))
range(10)
iter(range(10))
for i in range(10):

    print(i, end=' ')
N = 10 ** 12

for i in range(N):

    if i >= 10: break

    print(i, end=', ')
from itertools import count



for i in count():

    if i >= 10:

        break

    print(i, end=', ')
L = [2, 4, 6, 8, 10]

for i in range(len(L)):

    print(i, L[i])
for i, val in enumerate(L):

    print(i, val)
L = [2, 4, 6, 8, 10]

R = [3, 6, 9, 12, 15]

for lval, rval in zip(L, R):

    print(lval, rval)
# find the first 10 square numbers

square = lambda x: x ** 2

for val in map(square, range(10)):

    print(val, end=' ')
# find values up to 10 for which x % 2 is zero

is_even = lambda x: x % 2 == 0

for val in filter(is_even, range(10)):

    print(val, end=' ')
print(*range(10))
print(*map(lambda x: x ** 2, range(10)))
L1 = (1, 2, 3, 4)

L2 = ('a', 'b', 'c', 'd')
z = zip(L1, L2)

print(*z)
z = zip(L1, L2)

new_L1, new_L2 = zip(*z)

print(new_L1, new_L2)
from itertools import permutations

p = permutations(range(3))

print(*p)
from itertools import combinations

c = combinations(range(4), 2)

print(*c)
from itertools import product

p = product('ab', range(3))

print(*p)