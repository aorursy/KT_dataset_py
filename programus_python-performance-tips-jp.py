# just some preparation

from IPython.core import page

page.page = print
%%time

heavy_job = [x ** 3 for x in range(9999999)]
# without jupyter notebook

import timeit



def factorial(n):

    return n * factorial(n - 1) if n > 0 else 1



timeit.timeit('sum(map(factorial, range(500)))', globals = globals(), number = 10)
%%timeit -n 10



def factorial(n):

    return n * factorial(n - 1) if n > 0 else 1



sum(map(factorial, range(500)))
def relu(x):

    return x if x > 0 else 0



def test():

    result = [relu(x) for x in range(200000)]

    return result
# without jupyter notebook

import profile

profile.run('test()')
%prun test()
# !pip install line_profiler

%load_ext line_profiler
# without jupyter notebook

import line_profiler



lprofile = line_profiler.LineProfiler(test, relu)

lprofile.run('test()')

lprofile.print_stats()
%lprun -f test -f relu test()
import random

data = [i**2 for i in range(10000000)]

random.shuffle(data)
%%time

list_data = list(data)
%%time

set_data = set(data)
random.seed(0)
%%timeit -n 10

random.choice(data) in list_data
random.seed(0)
%%timeit -n 10

random.choice(data) in set_data
integers = [i for i in range(10000000)]

squares = [i * i for i in integers]

square_dict = {i: i * i for i in integers}
random.seed(0)
%%timeit -n 10

squares[integers.index(random.choice(integers))]
random.seed(0)
%%timeit -n 10

square_dict[random.choice(integers)]
%%time

i = 0

s = 0

while i < 1000000:

    s += i

    i += 1

s
%%time

s = 0

for i in range(1000000):

    s += i

s
%%time

s = sum(range(1000000))

s
some_list = [i * i for i in range(30000)]
%%time

result = [i / sum(some_list) for i in some_list]
%%time

total = sum(some_list)

result = [i / total for i in some_list]
%%time

def fib(n):

    return 1 if n in (1, 2) else fib(n - 1) + fib(n - 2)

fib(35)
%%time

def fib(n):

    a, b = 1, 1

    for i in range(n - 2):

        a, b = b, a + b

    return b

fib(35)
%%time

def fib(n):

    return 1 if n in (1, 2) else fib(n - 1) + fib(n - 2)

fib(35)
%%time

import functools

@functools.lru_cache(maxsize=50)

def fib(n):

    return 1 if n in (1, 2) else fib(n - 1) + fib(n - 2)

fib(35)
%%timeit -n 10

def square(x):

    return x * x



def square_sum(n):

    s = 0

    for i in range(1, n + 1):

        s += square(i)

    return s



square_sum(1000000)
%%timeit -n 10

import numba



@numba.jit

def square(x):

    return x * x



@numba.jit

def square_sum(n):

    s = 0

    for i in range(1, n + 1):

        s += square(i)

    return s



square_sum(1000000)
import collections

data = [x % 2020 for x in range(10000000)]
%%time

count_values = {}

for i in data:

    count = count_values.get(i, 0)

    count_values[i] = count + 1

count_values[10]
%%time

count_values = collections.Counter(data)

count_values[10]
dicts = [

    {hex(i * 16 + j): f'v{i}-{j}' for i in range(100000)}

    for j in range(16)

]
%%time

result = {}

for d in dicts:

    result.update(d)

len(result)
%%time

result = collections.ChainMap(*dicts)

len(result)
import numpy as np

N = 10000000
%%time

a = range(1, N, 3)

b = range(N, 1, -3)

result = [3 * aa - 2 * bb for aa, bb in zip(a, b)]
%%time

a = np.arange(1, N, 3)

b = np.arange(N, 1, -3)

result = 3 * a - 2 * b
import math

N = 10000000
%%time

a = range(1, N, 3)

b = [math.log(x) for x in a]
%%time

a = np.arange(1, N, 3)

b = np.log(a)
a = np.arange(-1000000, 1000000)
%%time

relu = np.vectorize(lambda x: x if x > 0 else 0)

result = relu(a)
%%time

relu = lambda x: np.where(x > 0, x, 0)

result = relu(a)
import pandas as pd

import numpy as np

df = pd.DataFrame(np.random.randint(-10, 11, size=(100000, 26)), columns=[chr(c) for c in range(ord('a'), ord('z') + 1)])

df
%%time

df.applymap(lambda x: np.sin(x) + np.cos(x))
%%time

df.apply(lambda x: np.sin(x) + np.cos(x))
%%time

np.sin(df) + np.cos(df)
%%time

df = pd.DataFrame(columns=[chr(c) for c in range(ord('a'), ord('z') + 1)])

for i in range(1000):

    df.loc[i, :] = range(i, i + 26)

df.shape
%%time

df = pd.DataFrame(np.zeros((1000, 26)), columns=[chr(c) for c in range(ord('a'), ord('z') + 1)])

for i in range(1000):

    df.loc[i, :] = range(i, i + 26)

df.shape
!pip install pandarallel

import pandarallel

import os

df = pd.DataFrame(np.random.randint(-10, 11, size=(100000, 26)), columns=[chr(c) for c in range(ord('a'), ord('z') + 1)])
%%time

df.apply(np.sum, axis=1)
%%time

pandarallel.pandarallel.initialize(nb_workers=os.cpu_count())

df.parallel_apply(np.sum, axis=1)
def busy(n):

    for _ in range(10000000):

        n * n

    return n * n



import multiprocessing
%%time

result = [busy(i) for i in range(10)]

result
%%time

with multiprocessing.Pool(os.cpu_count()) as pool:

    result = pool.map(busy, range(10))

result
import time

def io_busy(n):

    time.sleep(5)

    return n * n



import multiprocessing.pool
%%time

result = [io_busy(i) for i in range(4)]

result
%%time

with multiprocessing.pool.ThreadPool(os.cpu_count()) as pool:

    result = pool.map(io_busy, range(4))

result
%%time

with multiprocessing.pool.ThreadPool(os.cpu_count()) as pool:

    result = pool.map(busy, range(10))

result