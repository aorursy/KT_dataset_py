list(range(1, 11))
L = []

for x in range(1, 11):

    L.append(x * x)

    

L
[x * x for x in range(1, 11)]
[x * x for x in range(1, 11) if x % 2 == 0]
[m + n for m in 'ABC' for n in 'XYZ']
d = {'x': 'A', 'y': 'B', 'z': 'C' }

for k, v in d.items():

    print(k, '=', v)
d = {'x': 'A', 'y': 'B', 'z': 'C' }

[k + '=' + v for k, v in d.items()]
L = ['Hello', 'World', 'IBM', 'Apple']

[s.lower() for s in L]
L = [x * x for x in range(5)]

L
g = (x * x for x in range(5))

g
next(g)
next(g)
next(g)
next(g)
next(g)
next(g)
g = (x * x for x in range(5))
for e in g:

    print(e)
def fib(max):

    n, a, b = 0, 0, 1

    while n < max:

        print(b)

        a, b = b, a + b

        n = n + 1

    return 'done'
fib(5)
def fib(max):

    n, a, b = 0, 0, 1

    while n < max:

        yield b

        a, b = b, a + b

        n = n + 1

    return 'done'
f = fib(5)

f
def odd():

    print('step 1')

    yield 1

    print('step 2')

    yield(3)

    print('step 3')

    yield(5)
o = odd()
next(o)
next(o)
next(o)
next(o)
g = fib(5)
while True:

    try:

        x = next(g)

        print('g:',x)

    except StopIteration as e:

        print('Generator return value:', e.value)

        break
abs(-10)
abs
x = abs(-10)

x
f = abs

f
f = abs

f(-10)
abs = 10
abs(-10)
def add(x, y, f):

    return f(x) + f(y)
x = -5

y = 6

f = abs
add(x, y, f)
def calc_sum(*args):

    ax = 0

    for n in args:

        ax = ax + n

    return ax
def lazy_sum(*args):

    def sum():

        ax = 0

        for n in args:

            ax = ax + n

        return ax

    return sum
f = lazy_sum(1, 3, 5, 7, 9)

f
f()
f1 = lazy_sum(1, 3, 5, 7, 9)

f2 = lazy_sum(1, 3, 5, 7, 9)
f1 == f2
def count():

    fs = []

    for i in range(1, 4):

        def f():

             return i*i

        fs.append(f)

    return fs
f1, f2, f3 = count()
f1()
f2()
f3()
def count():

    def f(j):

        def g():

            return j*j

        return g

    fs = []

    for i in range(1, 4):

        fs.append(f(i)) # f(i)立刻被执行，因此i的当前值被传入f()

    return fs
f1, f2, f3 = count()
f1()
f2()
f3()
list(map(lambda x: x * x, [1, 2, 3, 4, 5, 6, 7, 8, 9]))
def f(x):

    return x * x
f = lambda x: x * x

f
f(5)
def build(x, y):

    return lambda: x * x + y * y
def now():

    print('2020-01-25')
f = now
f()
now.__name__
f.__name__
def log(func):

    def wrapper(*args, **kw):

        print('call %s():' % func.__name__)

        return func(*args, **kw)

    return wrapper
@log

def now():

    print('2015-3-25')
now()
int('12345')
int('12345', base=8)
int('12345', base=16)
def int2(x, base=2):

    return int(x, base)
int2('1000000')
int2('1010101')
import functools

int2 = functools.partial(int, base=2)
int2('1000000')
int2('1010101')
int2('1000000', base=10)
int2 = functools.partial(int, base=2)
int2('10010')
kw = { 'base': 2 }

int('10010', **kw)
max2 = functools.partial(max, 10)
max2(5, 6, 7)
args = (10, 5, 6, 7)

max(*args)