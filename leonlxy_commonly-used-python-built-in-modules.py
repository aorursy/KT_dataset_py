from datetime import datetime

now = datetime.now() 

print(now)
print(type(now))
dt = datetime(2020, 1, 19, 12, 20)

print(dt)
dt = datetime(2020, 1, 19, 12, 20) 

dt.timestamp() 
t = 1579404000.0

print(datetime.fromtimestamp(t))
t = 1579404000.0

print(datetime.fromtimestamp(t)) # local time
print(datetime.utcfromtimestamp(t)) # UTC time
cday = datetime.strptime('2020-01-19 12:20:00', '%Y-%m-%d %H:%M:%S')

print(cday)
now = datetime.now()

print(now.strftime('%a, %b %d %H:%M'))
from datetime import timedelta
now = datetime.now()

now
now + timedelta(hours=10)
now - timedelta(days=1)
now + timedelta(days=2, hours=12)
p = (1, 2)
from collections import namedtuple



Point = namedtuple('Point', ['x', 'y'])

p = Point(1, 2)
p.x
p.y
isinstance(p, Point)
isinstance(p, tuple)
# namedtuple('name', [Attribute(type:list)]):

Circle = namedtuple('Circle', ['x', 'y', 'r'])
from collections import deque



q = deque(['a', 'b', 'c'])

q.append('x')

q.appendleft('y')

q
from collections import defaultdict



dd = defaultdict(lambda: 'N/A')

dd['key1'] = 'abc'
dd['key1'] # key1 exist
dd['key2'] # key2 doesn't exist, return default value
from collections import Counter



c = Counter()

for ch in 'programming':

    c[ch] = c[ch] + 1

    

c
c.update('hello')

c
import hashlib



md5 = hashlib.md5()

md5.update('how to use md5 in python hashlib?'.encode('utf-8'))

print(md5.hexdigest())
md5 = hashlib.md5()

md5.update('how to use md5 in '.encode('utf-8'))

md5.update('python hashlib?'.encode('utf-8'))

print(md5.hexdigest())
md5 = hashlib.md5()

md5.update('how to use md0 in python hashlib?'.encode('utf-8'))

print(md5.hexdigest())
sha1 = hashlib.sha1()

sha1.update('how to use sha1 in '.encode('utf-8'))

sha1.update('python hashlib?'.encode('utf-8'))

print(sha1.hexdigest())
def calc_md5(password):

    return get_md5(password + 'the-Salt')
import itertools



natuals = itertools.count(1)

# for n in natuals:

#     print(n)
cs = itertools.cycle('ABC') # String is also a sequence.

# for c in cs:

#     print(c)
ns = itertools.repeat('A', 3)
for n in ns:

    print(n)
natuals = itertools.count(1)

ns = itertools.takewhile(lambda x: x <= 10, natuals)

list(ns)
for c in itertools.chain('ABC', 'XYZ'):

    print(c)
for key, group in itertools.groupby('AAABBBCCAAA'):

    print(key, list(group))
for key, group in itertools.groupby('AaaBBbcCAAa', lambda c: c.upper()):

    print(key, list(group))