!pip install PEQNP
import peqnp as pn



rsa = 3007



pn.engine(rsa.bit_length())



p = pn.integer()

q = pn.integer()



assert p * q == rsa



while pn.satisfy():

    print(p, q)
import time

import random

import peqnp as pn





def generator(n, max_val):

    return sorted([random.randint(1, max_val) for _ in range(n)])





def differences(lst):

    return [abs(lst[i] - lst[i - 1]) for i in range(1, len(lst))]





# 100 tests

for n in range(1, 10):



    m = random.randint(1, n ** 2)



    original = generator(n, m)

    diffs = differences(original)



    print('N, M         : {}, {}'.format(n, m))

    print('DIFFERENCES  : {}'.format(diffs))

    print('ORIGINAL     : {}'.format(original))



    # only one tip

    ith = random.choice(range(n))

    tip = original[ith]



    # init timer

    ini = time.time()



    # Empirical bits necessarily to solve the problem.

    pn.engine(sum(diffs).bit_length() + 4)



    # Declare a n-vector of integer variables to store the solution.

    x = pn.vector(size=n)



    # The tip is on x at index ith

    assert tip == pn.index(ith, x)



    # The i-th element of the instance is the absolute difference of two consecutive elements

    for i in range(n - 1):

        assert x[i] <= x[i + 1]

        assert pn.index(i, diffs) == x[i + 1] - x[i]



    # Solve the problem for only one solution

    # Turbo parameter is a destructive simplification

    # Solve with all power os SLIME SAT Solver but only for the fist solution.

    if pn.satisfy(turbo=True):

        o = [abs(x[i + 1] - x[i]) for i in range(n - 1)]

        c = 100 * len(set(map(int, x)).intersection(set(original))) / len(set(original))

        print('SOLVED       : {}'.format(x))

        print('COINCIDENCES : {}%'.format(c))

        if o == diffs:

            print('OK! - {}s'.format(time.time() - ini))

        else:

            print('NOK! - {}s'.format(time.time() - ini))

            raise Exception('ERROR!')

        if c != 100:

            raise Exception('Hypothesis Fail - 100%')
import peqnp as pn



pn.engine()



x0 = pn.linear(is_real=True)

x1 = pn.linear(is_real=True)

x2 = pn.linear(is_real=True)

x3 = pn.linear(is_real=True)

x4 = pn.linear(is_real=True)

x5 = pn.linear()

x6 = pn.linear()

x7 = pn.linear()

x8 = pn.linear()

x9 = pn.linear()

assert +6.4160 * x0 + 20.3590 * x1 + 1.5981 * x2 - 1.2071 * x3 - 4.6026 * x4 - 5.7098 * x5 - 4.1160 * x6 + 1.2467 * x7 - 14.2028 * x8 + 6.0885 * x9 <= 51.0000

assert -0.1930 * x0 + 1.1859 * x1 + 2.9537 * x2 - 2.3777 * x3 + 1.4154 * x4 + 9.2526 * x5 - 3.6259 * x6 + 3.4193 * x7 - 21.4218 * x8 - 0.7692 * x9 <= 41.0000

assert -27.1712 * x0 - 21.2901 * x1 + 32.6104 * x2 + 1.4699 * x3 + 8.1651 * x4 - 12.8153 * x5 + 2.4100 * x6 - 57.0053 * x7 - 7.2989 * x8 + 7.0098 * x9 <= 79.0000

assert -2.3318 * x0 + 0.8284 * x1 + 6.2896 * x2 + 0.6104 * x3 - 31.1931 * x4 + 4.1556 * x5 + 2.6317 * x6 - 48.5799 * x7 - 1.1840 * x8 + 28.7408 * x9 <= 93.0000

assert +12.0876 * x0 + 1.2307 * x1 - 0.9757 * x2 - 4.2857 * x3 + 4.8579 * x4 + 19.5823 * x5 + 18.5408 * x6 - 3.0287 * x7 + 2.0617 * x8 - 3.5956 * x9 <= 25.0000

assert -50.3777 * x0 + 6.9980 * x1 - 67.9637 * x2 - 2.0244 * x3 + 7.8885 * x4 - 2.5394 * x5 - 5.3325 * x6 + 0.3273 * x7 - 249.6093 * x8 + 3.7692 * x9 <= 41.0000

assert +43.2031 * x0 - 2.0964 * x1 + 10.1320 * x2 - 13.9120 * x3 + 3.2838 * x4 + 10.6522 * x5 + 6.2647 * x6 + 2.8932 * x7 - 6.3529 * x8 + 20.0324 * x9 <= 78.0000

assert -2.0752 * x0 - 7.4701 * x1 - 0.2348 * x2 - 2.0003 * x3 - 0.6376 * x4 + 1.7804 * x5 + 119.5958 * x6 - 6.2943 * x7 + 3.3538 * x8 - 2.6467 * x9 <= 27.0000

assert +3.1615 * x0 + 6.0781 * x1 - 1.8893 * x2 - 3.2409 * x3 - 34.0146 * x4 + 23.8191 * x5 - 8.8890 * x6 - 6.8173 * x7 + 6.7114 * x8 - 8.1344 * x9 <= 21.0000

assert +0.0000 * x0 + 13.1440 * x1 + 7.5737 * x2 + 2.8277 * x3 - 4.3930 * x4 + 0.0000 * x5 - 22.1786 * x6 + 2.8980 * x7 - 9.0440 * x8 - 60.4170 * x9 <= 93.0000

assert x0 <= 92.0000

assert x1 <= 46.0000

assert x2 <= 74.0000

assert x3 <= 78.0000

assert x4 <= 41.0000

assert x5 <= 47.0000

assert x6 <= 33.0000

assert x7 <= 35.0000

assert x8 <= 23.0000

assert x9 <= 63.0000

print(pn.maximize(+0.0000 * x0 + 9.6856 * x1 + 0.0000 * x2 - 7.8267 * x3 - 3.4649 * x4 - 6.3391 * x5 - 3.6316 * x6 + 44.7655 * x7 + 3.7610 * x8 - 57.1083 * x9))

print(x0)

print(x1)

print(x2)

print(x3)

print(x4)

print(x5)

print(x6)

print(x7)

print(x8)

print(x9)
import peqnp as pn



pn.engine(10)



x = pn.integer()

y = pn.integer()



assert x ** 3 - x + 1 == y ** 2



assert x != 0

assert y != 0



while pn.satisfy():

    print('{0} ** 3 - {0} + 1, {1} ** 2'.format(x, y))
import peqnp as pn



pn.engine(10)



x = pn.rational()

y = pn.rational()



assert x ** 3 + x * y == y ** 2



while pn.satisfy():

    print('{0} ** 3 + {0} * {1} == {1} ** 2'.format(x, y))
import peqnp as pn



pn.engine(11)



x = pn.gaussian()

y = pn.gaussian()



assert x ** 3 + x + 1 == y ** 2



while pn.satisfy():

    print('{0} ** 3 + {0} + 1 == {1} ** 2'.format(complex(x), complex(y)))
import peqnp as pn



pn.engine(10)



x = pn.vector(size=2, is_gaussian=True)

y = pn.gaussian()



assert sum(x) ** 3 == y ** 5

assert y != complex(0, 0)



while pn.satisfy():

    print('sum({0}) ** 3 == {1} ** 5'.format(x, y))
import numpy as np

import peqnp as pn

import matplotlib.pyplot as plt



dim = 2



pn.engine(10)



ps = pn.vector(size=dim, is_rational=True)



assert sum([p ** dim for p in ps]) <= 1



dots = []

while pn.satisfy():

    dots.append(np.vectorize(float)(ps))



x, y = zip(*dots)

plt.axis('equal')

plt.plot(x, y, 'r.')

plt.show()
import functools

import operator



import peqnp as pn



n, m, cnf = 10, 24, [[9, -5, 10, -6, 3],

                     [6, 8],

                     [8, 4],

                     [-10, 5],

                     [-9, 8],

                     [-9, -3],

                     [-2, 5],

                     [6, 4],

                     [-2, -1],

                     [7, -2],

                     [-9, 4],

                     [-1, -10],

                     [-3, 4],

                     [7, 5],

                     [6, -3],

                     [-10, 7],

                     [-1, 7],

                     [8, -3],

                     [-2, -10],

                     [-1, 5],

                     [-7, 1, 9, -6, 3],

                     [-9, 6],

                     [-8, 10, -5, -4, 2],

                     [-4, -7, 1, -8, 2]]



pn.engine(bits=1)

x = pn.tensor(dimensions=(n,), key='x')

assert functools.reduce(operator.iand, (functools.reduce(operator.ior, (x[[abs(lit) - 1]](lit < 0, lit > 0) for lit in cls)) for cls in cnf)) == 1

if pn.satisfy(turbo=True):

    print('SAT')

    print(' '.join(map(str, [(i + 1) if b else -(i + 1) for i, b in enumerate(x.binary)])) + ' 0')

else:

    print('UNSAT')
import peqnp as pn



# Ths bits of the clique to search

k = 3



# Get the graph, and the dimension for the graph

n, matrix = 5, [(1, 0), (0, 2), (1, 4), (2, 1), (4, 2), (3, 2)]



# Ensure the problem can be represented

pn.engine(bits=k.bit_length())



# Declare an integer of n-bits

bits = pn.integer(bits=n)



# The bits integer have "bits"-active bits, i.e, the clique has "bits"-elements

assert sum(pn.switch(bits, i) for i in range(n)) == k



# This entangles all elements that are joined together

for i in range(n - 1):

    for j in range(i + 1, n):

        if (i, j) not in matrix and (j, i) not in matrix:

            assert pn.switch(bits, i) + pn.switch(bits, j) <= 1



if pn.satisfy(turbo=True):

    print(k)

    print(' '.join([str(i) for i in range(n) if not bits.binary[i]]))

else:

    print('Infeasible ...')
import peqnp as pn



# Get the graph and dimension, and the bits of the cover.

n, graph, vertex, k = 5, [(1, 0), (0, 2), (1, 4), (2, 1), (4, 2), (3, 2)], [0, 1, 2, 3, 4], 3



# Ensure the problem can be represented

pn.engine(bits=n.bit_length() + 1)



# An integer with n-bits to store the indexes for the cover

index = pn.integer(bits=n)



# This entangled the all possible covers

for i, j in graph:

    assert pn.switch(index, vertex.index(i), neg=True) + pn.switch(index, vertex.index(j), neg=True) >= 1



# Ensure the cover has bits k

assert sum(pn.switch(index, vertex.index(i), neg=True) for i in vertex) == k



if pn.satisfy(turbo=True):

    opt = sum(index.binary)

    print('p bits {}'.format(opt))

    print(' '.join([str(vertex[i]) for i in range(n) if index.binary[i]]))

else:

    print('Infeasible ...')
import numpy as np

import peqnp as pn



n = 6

m = 3



pn.engine(n.bit_length())



Y = pn.vector(size=n ** m)



pn.apply_single(Y, lambda k: k < n)



Y = np.reshape(Y, newshape=(m * [n]))



for i in range(n):

    pn.all_different(Y[i])

    pn.all_different(Y.T[i])

    for j in range(n):

        pn.all_different(Y[i][j])

        pn.all_different(Y.T[i][j])



for idx in pn.hyper_loop(m - 1, n):

    s = Y

    for i in idx:

        s = s[i]

        pn.all_different(s)

        pn.all_different(s.T)



if pn.satisfy(turbo=True):

    y = np.vectorize(int)(Y).reshape(m * [n])

    print(y)

else:

    print('Infeasible ...')
import peqnp as pn

import numpy as np



n = 100

data = np.random.logistic(size=(n, 2))

seq = pn.hess_sequence(n, oracle=lambda seq: sum(np.linalg.norm(data[seq[i - 1]] - data[seq[i]]) for i in range(n)), fast=False)

x, y = zip(*[data[i] for i in seq + [seq[0]]])

plt.plot(x, y, 'k-')

plt.plot(x, y, 'r.')

plt.show()
import peqnp as pn

import numpy as np



n = 10

values = np.random.logistic(size=n)

profit = np.random.logistic(size=n)

capacity = np.random.sample()

pn.engine()

selects = pn.vector(size=n, is_mip=True)

pn.apply_single(selects, lambda x: x <= 1)

assert np.dot(values, selects) <= capacity

opt = pn.maximize(np.dot(profit, selects))

slots = list(map(int, selects))

print('PROFIT  : {} vs {}'.format(np.dot(profit, slots), opt))

print('VALUES  : {} <= {}'.format(np.dot(values, slots), capacity))

print('SELECT  : {}'.format(slots))
import peqnp as pn

import numpy as np



n = 3

pn.engine(5)

c = pn.integer()

xs = pn.matrix(dimensions=(n, n))

pn.apply_single(pn.flatten(xs), lambda x: x > 0)

pn.all_different(pn.flatten(xs))

for i in range(n):

    assert sum(xs[i][j] for j in range(n)) == c

for j in range(n):

    assert sum(xs[i][j] for i in range(n)) == c

assert sum(xs[i][i] for i in range(n)) == c

assert sum(xs[i][n - 1 - i] for i in range(n)) == c

if pn.satisfy(turbo=True):

    print(c)

    print(np.vectorize(int)(xs))

else:

    print('Infeasible ...')
import peqnp as pn

import numpy as np



bits = 7

size = 3 * 10

triplets = []

while len(triplets) < size:

    a = np.random.randint(1, 2 ** bits)

    b = np.random.randint(1, 2 ** bits)

    if a != b and a not in triplets and b not in triplets and a + b not in triplets:

        triplets += [a, b, a + b]

triplets.sort()

print(triplets)

pn.engine(bits=max(triplets).bit_length())

xs, ys = pn.permutations(triplets, size)

for i in range(0, size, 3):

    assert ys[i] + ys[i + 1] == ys[i + 2]

if pn.satisfy(turbo=True):

    for i in range(0, size, 3):

        print('{} == {} + {}'.format(ys[i + 2], ys[i], ys[i + 1]))

else:

    print('Infeasible ...')
import peqnp as pn

import numpy as np



universe = np.random.randint(1, 1000, size=32)

t = np.random.randint(min(universe), sum(universe))

print(t, universe)

pn.engine(t.bit_length())

bits, subset = pn.subsets(universe)

assert sum(subset) == t

if pn.satisfy(turbo=True):

    solution = [universe[i] for i in range(len(universe)) if bits.binary[i]]

    print(sum(solution), solution)

else:

    print('Infeasible ...')
import peqnp as pn

import numpy as np



universe = np.random.randint(1, 1000, size=32)

t = np.random.randint(min(universe), sum(universe))

print(t, universe)

pn.engine(t.bit_length())

T = pn.tensor(dimensions=(len(universe)))

assert sum(T[[i]](0, universe[i]) for i in range(len(universe))) == t

if pn.satisfy(turbo=True):

    solution = [universe[i] for i in range(len(universe)) if T.binary[i]]

    print(sum(solution), solution)

else:

    print('Infeasible ...')
import peqnp as pn

import numpy as np



def gen_instance(n):

    import random

    y = list(range(1, n + 1))

    random.shuffle(y)

    return [abs(y[i + 1] - y[i]) for i in range(n - 1)]





import time

start = time.time()

times = []

sizes = []

for n in range(1, 30):

    diffs = gen_instance(n)

    ini = time.time()

    pn.engine(n.bit_length() + 1)

    x = pn.vector(size=n)

    pn.all_different(x)

    pn.apply_single(x, lambda a: 1 <= a <= n)

    for i in range(n - 1):

        assert pn.index(i, diffs) == pn.one_of([x[i + 1] - x[i], x[i] - x[i + 1]])

    if pn.satisfy(turbo=True):

        end = time.time() - ini

        xx = [abs(x[i + 1] - x[i]) for i in range(n - 1)]

        if xx == diffs:

            sizes.append(n)

            times.append(end)

        else:

            raise Exception('Error!')

    else:

        raise Exception('Error!')

end = time.time() - start

plt.title('TIME {}(s)'.format(end))

plt.plot(sizes, times, 'k-')

plt.plot(sizes, times, 'r.')

plt.show()

plt.close()
import sys



import peqnp as pn

import numpy as np





n = 10

M = np.random.randint(0, 2, size=(n, n))

print(M)

pn.engine((n ** 2).bit_length())

ids, elements = pn.matrix_permutation((1 - M).flatten(), n)

assert sum(elements) == 0

if pn.satisfy(turbo=True):

    for i in ids:

        for j in ids:

            sys.stdout.write('{} '.format(M[i.value][j.value]))

        sys.stdout.write('\n') 

    sys.stdout.write('\n')

else:

    print('Infeasible ...')
import peqnp as pn

import numpy as np



capacity = 50

size = 50

elements = sorted([np.random.randint(1, capacity // 2 - 1) for _ in range(size)], reverse=True)

print(capacity)

print(elements)

bins = int(np.ceil(sum(elements) / capacity))

while True:

    pn.engine(bits=capacity.bit_length() + 1)

    slots = pn.vector(bits=len(elements), size=bins)

    for i in range(len(elements)):

        assert sum(pn.switch(slot, i) for slot in slots) == 1

    for slot in slots:

        assert sum(pn.switch(slot, i) * elements[i] for i in range(len(elements))) <= capacity

    if pn.satisfy(turbo=True):

        print('Solution for {} bins...'.format(bins))

        for slot in slots:

            print(''.join(['_' if boolean else '#' for boolean in slot.binary]))

        for slot in slots:

            sub = [item for i, item in enumerate(elements) if not slot.binary[i]]

            print(sum(sub), sub)

        break

    else:

        print('No solution for {} bins...'.format(bins))

        bins += 1
import peqnp as pn

import numpy as np



n, m = 10, 5

cc = np.random.randint(0, 1000, size=(n, m))

d = np.dot(cc, np.random.randint(0, 2, size=(m,)))

print(cc)

print(d)

pn.engine(bits=int(np.sum(cc)).bit_length())

xs = pn.vector(size=m)

pn.all_binaries(xs)

assert (np.dot(cc, xs) == d).all()

if pn.satisfy():

    print(xs)

    print('Proof:')

    print(np.dot(cc, xs))

else:

    print('Infeasible...')
import peqnp as pn





def completion(n, m, seed):

    import random

    """

    http://www.csplib.org/Problems/prob079/data/queens-gen-fast.py.html

    """

    random.seed(seed)



    d1 = [0 for _ in range(2 * n - 1)]

    d2 = [0 for _ in range(2 * n - 1)]



    valid_rows = [i for i in range(n)]

    valid_cols = [j for j in range(n)]



    def no_attack(r, c):

        return d1[r + c] == 0 and d2[r - c + n - 1] == 0



    pc = []

    queens_left = n



    for attempt in range(n * n):

        i = random.randrange(queens_left)

        j = random.randrange(queens_left)

        r = valid_rows[i]

        c = valid_cols[j]

        if no_attack(r, c):

            pc.append([r, c])

            d1[r + c] = 1

            d2[r - c + n - 1] = 1

            valid_rows[i] = valid_rows[queens_left - 1]

            valid_cols[j] = valid_cols[queens_left - 1]

            queens_left -= 1

            if len(pc) == m:

                return [[x + 1, y + 1] for x, y in pc]



def show(pc):

    table = ''

    for i in range(1, n + 1):

        table += ''

        for j in range(1, n + 1):

            if [i, j] not in pc:

                table += '. '

            else:

                table += 'Q '

        table += '\n'

    print(table)

    print('# seed = {}'.format(seed))

    

n, m, seed = 30, 15, 0

placed_queens = completion(n, m, seed)

show(placed_queens)

pn.engine(bits=n.bit_length() + 1)

qs = pn.vector(size=n)

for (a, b) in placed_queens:

    assert qs[a - 1] == b - 1

pn.apply_single(qs, lambda x: x < n)

pn.apply_dual(qs, lambda x, y: x != y)

pn.apply_dual([qs[i] + i for i in range(n)], lambda x, y: x != y)

pn.apply_dual([qs[i] - i for i in range(n)], lambda x, y: x != y)

if pn.satisfy(turbo=True):

    for i in range(n):

        print(''.join(['Q ' if qs[i] == j else '. ' for j in range(n)]))

    print('')

else:

    print('Infeasible ...')
import numpy as np

import peqnp as pn



n = 90

m = 5



z = 11191

c = [360, 83, 59, 130, 431, 67, 230, 52, 93, 125, 670, 892, 600, 38, 48, 147, 78, 256, 63, 17, 120, 164, 432, 35, 92, 110, 22, 42, 50, 323, 514, 28, 87, 73, 78, 15, 26, 78, 210, 36, 85, 189, 274, 43, 33, 10, 19, 389, 276, 312, 94, 68, 73, 192, 41, 163, 16, 40, 195, 138, 73, 152, 400, 26, 14, 170, 205, 57, 369, 435, 123, 25, 94, 88, 90, 146, 55, 29, 82, 74, 100, 72, 31, 29, 316, 244, 70, 82, 90, 52]



b = [2100, 1100, 3300, 3700, 3600]

a = [[7, 0, 30, 22, 80, 94, 11, 81, 70, 64, 59, 18, 0, 36, 3, 8, 15, 42, 9, 0, 42, 47, 52, 32, 26, 48, 55, 6, 29, 84, 2, 4, 18, 56, 7, 29, 93, 44, 71, 3, 86, 66, 31, 65, 0, 79, 20, 65, 52, 13, 48, 14, 5, 72, 14, 39, 46, 27, 11, 91, 15, 25, 0, 94, 53, 48, 27, 99, 6, 17, 69, 43, 0, 57, 7, 21, 78, 10, 37, 26, 20, 8, 4, 43, 17, 25, 36, 60, 84, 40],

     [8, 66, 98, 50, 0, 30, 0, 88, 15, 37, 26, 72, 61, 57, 17, 27, 83, 3, 9, 66, 97, 42, 2, 44, 71, 11, 25, 74, 90, 20, 0, 38, 33, 14, 9, 23, 12, 58, 6, 14, 78, 0, 12, 99, 84, 31, 16, 7, 33, 20, 5, 18, 96, 63, 31, 0, 70, 4, 66, 9, 15, 25, 2, 0, 48, 1, 40, 31, 82, 79, 56, 34, 3, 19, 52, 36, 95, 6, 35, 34, 74, 26, 10, 85, 63, 31, 22, 9, 92, 18],

     [3, 74, 88, 50, 55, 19, 0, 6, 30, 62, 17, 81, 25, 46, 67, 28, 36, 8, 1, 52, 19, 37, 27, 62, 39, 84, 16, 14, 21, 5, 60, 82, 72, 89, 16, 5, 29, 7, 80, 97, 41, 46, 15, 92, 51, 76, 57, 90, 10, 37, 25, 93, 5, 39, 0, 97, 6, 96, 2, 81, 69, 4, 32, 78, 65, 83, 62, 89, 45, 53, 52, 76, 72, 23, 89, 48, 41, 1, 27, 19, 3, 32, 82, 20, 2, 51, 18, 42, 4, 26],

     [21, 40, 0, 6, 82, 91, 43, 30, 62, 91, 10, 41, 12, 4, 80, 77, 98, 50, 78, 35, 7, 1, 96, 67, 85, 4, 23, 38, 2, 57, 4, 53, 0, 33, 2, 25, 14, 97, 87, 42, 15, 65, 19, 83, 67, 70, 80, 39, 9, 5, 41, 31, 36, 15, 30, 87, 28, 13, 40, 0, 51, 79, 75, 43, 91, 60, 24, 18, 85, 83, 3, 85, 2, 5, 51, 63, 52, 85, 17, 62, 7, 86, 48, 2, 1, 15, 74, 80, 57, 16],

     [94, 86, 80, 92, 31, 17, 65, 51, 46, 66, 44, 3, 26, 0, 39, 20, 11, 6, 55, 70, 11, 75, 82, 35, 47, 99, 5, 14, 23, 38, 94, 66, 64, 27, 77, 50, 28, 25, 61, 10, 30, 15, 12, 24, 90, 25, 39, 47, 98, 83, 56, 36, 6, 66, 89, 45, 38, 1, 18, 88, 19, 39, 20, 1, 7, 34, 68, 32, 31, 58, 41, 99, 92, 67, 33, 26, 25, 68, 37, 6, 11, 17, 48, 79, 63, 77, 17, 29, 18, 60]]





pn.engine()

xs = np.asarray(pn.vector(size=n, is_mip=True))

pn.apply_single(xs, lambda x: x <= 1)

for i in range(m):

    assert np.dot(a[i], xs) <= b[i]

assert np.dot(c, xs) <= z

print(pn.maximize(np.dot(c, xs)))

print(xs)
import numpy as np

import peqnp as pn



size = 20



data = np.random.randint(1000, size=size)



print(data)



pn.engine(int(sum(data)).bit_length())



T, sub, com = pn.subsets(data, complement=True)



assert sum(sub) == sum(com)



if pn.satisfy():

    sub_ = [data[i] for i in range(size) if T.binary[i]]

    com_ = [data[i] for i in range(size) if not T.binary[i]]

    print(sum(sub_), sub_)

    print(sum(com_), com_)

else:

    print('Infeasible ...')
import numpy as np

import peqnp as pn

import matplotlib.pyplot as plt





def plot(I, J=None, X=None, title='Original', obj=0):

    plt.figure(figsize=(10, 10))

    plt.title('{} : {}'.format(title, obj))

    a, b = zip(*I)

    plt.scatter(a, b, c='blue', s=50, alpha=0.6)

    if J is not None:

        if X is not None:

            for i in range(m):

                for j in range(n):

                    if X[i][j]:

                        plt.plot([I[i][0], J[j][0]], [I[i][1], J[j][1]], 'g-',alpha=0.2)

        a, b = zip(*J)

        plt.scatter(a, b, c='red', s=300, alpha=0.8)

    else:

        a, b = zip(*J)

        plt.scatter(a, b, c='red', s=300, alpha=0.8)

    plt.show()

    plt.close()





def oracle(seq):

    global O, glb, n

    M = np.zeros(shape=(m, n))

    for i in range(m):

        for j in range(n):

            M[i][j] = np.linalg.norm(I[i] - J[seq[j]])

    pn.engine()

    X = np.asarray(pn.matrix(dimensions=(m, n), is_mip=True))

    pn.all_binaries(X.flatten())

    assert sum(X.flatten()) == m

    assert (X.sum(axis=1) == 1).all()

    obj = pn.minimize(sum(X[i][j] * M[i][j] for i in range(m) for j in range(n)))

    O = np.vectorize(int)(X)

    return obj



m = 50

k = 15

n = 3

I = np.random.sample(size=(m, 2))    

J = np.random.sample(size=(k, 2))

plot(I, J)

seq = pn.hess_sequence(k, oracle=oracle, fast=False)

plot(I, J[seq][:n], O, 'www.peqnp.com', oracle(seq))
import numpy as np

import peqnp as pn

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs





def plot(X=None):

    fig, ax = plt.subplots(figsize=(10, 10))

    plt.title('PEQNP\nwww.peqnp.com')

    plt.scatter(N[:, 0], N[:, 1], c='b', s=P * 500, alpha=0.5)

    for i in range(m):

        ax.annotate(str(round(P[i], 2)), (N[:, 0][i], N[:, 1][i]), size=10)

    plt.scatter(M[:, 0], M[:, 1], c='r', s=L * 500, alpha=0.7)

    for i in range(n):

        ax.annotate(str(round(T[i], 2)), (M[:, 0][i] + 0.1, M[:, 1][i] + 0.1))

        ax.annotate(str(L[i]), (M[:, 0][i] - 0.1, M[:, 1][i] - 0.1))

    if X is not None:

        for i in range(m):

            for j in range(n):

                if X[i][j]:

                    plt.plot([N[i][0], M[j][0]], [N[i][1], M[j][1]], 'k--', alpha=0.3)

    plt.show()





n = 4



L = np.random.randint(1, 100, size=n)  # capacity x facilities



m = sum(L)



N, _ = make_blobs(n_samples=m)  # customers

P = np.random.sample(size=m)  # priorities x customers

M = np.random.normal(size=(n, 2)) * n  # facilities

T = np.random.sample(size=n)  # priorities x facility

C = np.zeros(shape=(m, n))

for i in range(m):

    for j in range(n):

        C[i][j] = np.linalg.norm(N[i] - M[j])



D = np.zeros(shape=(m, n))

for i in range(m):

    for j in range(n):

        D[i][j] = P[i] - T[j]



plot()



pn.engine()

X = np.asarray(pn.matrix(dimensions=(m, n), is_mip=True))

pn.all_binaries(X.flatten())

assert (X.sum(axis=0) <= L).all()

assert (X.sum(axis=1) == 1).all()

print(pn.minimize((X * C * D).sum()))

plot(np.vectorize(int)(X))
import numpy as np

import peqnp as pn





def expand_line(line):

    return line[0] + line[5:9].join([line[1:5] * (base - 1)] * base) + line[9:13]





def show(board):

    import string

    line0 = expand_line('╔═══╤═══╦═══╗')

    line1 = expand_line('║ . │ . ║ . ║')

    line2 = expand_line('╟───┼───╫───╢')

    line3 = expand_line('╠═══╪═══╬═══╣')

    line4 = expand_line('╚═══╧═══╩═══╝')



    symbol = ' ' + string.printable.replace(' ', '')

    nums = [[''] + [symbol[n] for n in row] for row in board]

    print(line0)

    for r in range(1, side + 1):

        print("".join(n + s for n, s in zip(nums[r - 1], line1.split('.'))))

        print([line2, line3, line4][(r % side == 0) + (r % base == 0)])





def generate(base):

    # pattern for a baseline valid solution

    def pattern(r, c):

        return (base * (r % base) + r // base + c) % side



    # randomize rows, columns and numbers (of valid base pattern)

    from random import sample



    def shuffle(s):

        return sample(s, len(s))



    rBase = range(base)

    rows = [g * base + r for g in shuffle(rBase) for r in shuffle(rBase)]

    cols = [g * base + c for g in shuffle(rBase) for c in shuffle(rBase)]

    nums = shuffle(range(1, base * base + 1))



    # produce board using randomized baseline pattern

    board = [[nums[pattern(r, c)] for c in cols] for r in rows]



    squares = side * side

    empties = (squares * 3) // 4

    for p in map(int, sample(range(squares), empties)):

        board[p // side][p % side] = 0



    show(board)

    return board





base = 4

side = base * base



puzzle = np.asarray(generate(base))



pn.engine(side.bit_length())



board = np.asarray(pn.matrix(dimensions=(side, side)))

pn.apply_single(board.flatten(), lambda x: 1 <= x <= side)



for i in range(side):

    for j in range(side):

        if puzzle[i][j]:

            assert board[i][j] == puzzle[i][j]



for c, r in zip(board, board.T):

    pn.all_different(c)

    pn.all_different(r)



for i in range(base):

    for j in range(base):

        pn.all_different(board[i * base:(i + 1) * base, j * base:(j + 1) * base].flatten())



if pn.satisfy(turbo=True):

    show(np.vectorize(int)(board))
