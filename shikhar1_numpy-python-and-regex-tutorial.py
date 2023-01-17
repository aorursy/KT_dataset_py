import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dask.dataframe as dd
import itertools
a, b, c = [1, 2, 3]
a, b, c
l = [1, 2, 3, 4, 5]
l[-2]
l[-3:-1]
a = ['cat', 'dog', 'rat']
"".join(a)
",".join(a)
" ".join(a)
'a b c'.split()  #default space
'a b c'.split(" ")
'a b c'.split()  #default space
'cat,dog,rat'.split(',')
l[::-1]  #reversing the list
"cat" * 4
s = 'abcbde'
s[::-1]
for i, e in enumerate(l):
    print(i, e)
dictionary = {'spain': 'madrid', 'france': 'paris'}
for key, value in dictionary.items():
    print(key, " : ", value)
print('')
a = [1, 2, 3]
b = ['a', 'b', 'c']
c = [6, 7, 8]
z = zip(a, b, c)
for e in z:
    print(e)
dictionary
dict(zip(dictionary.values(), dictionary.keys()))
a = [[1, 2], [3, 4], [5, 6, 7]]
list(itertools.chain.from_iterable(a))
sum(a, [])
m = {i: i * 4 for i in range(10)}
m
dictionary
{v: k for k, v in dictionary.items()}
a = [1, 2, 3]
it = iter(a)  #creates a iterable over the list
next(it)  #calls next iteration
print(*it)  #print remaining iterations
it = iter(a)  #creates a iterable over the list
next(it)  #calls next iteration
next(it)
next(it)
next(it)
a = "abcdef"
it = iter(a)
next(it)
print(*it)  #print remaining iterations
a = {"a": 1, "B": 3, "c": 3}
it = iter(a)  #by deafult it creates iterable over the keys
next(it)
l1 = [1, 2, 3, 3, 5, 6, 6, 7]
l2 = [1, 2, 4, 2]
s1 = set(l1)
s2 = set(l2)
s1
s2
s1.intersection(s2)
s1.union(s2)
s1.difference(s2)
s2.difference(s1)
l1
l1.pop()  #this will pop the last element from the list
l1  #we can see that the element is no longer in the list
l1.pop(3)  #we can sepecify the index of the element for popping too
l1
l1, l2
l1.extend(l2)
l1
m = dict()
m['a']
import collections
m = collections.defaultdict(int)
m['a']
m = collections.defaultdict(lambda: 2)
m['a']
for p in itertools.product([1, 2, 3], [4, 5]):
    print(p)
x = 5


def f():
    y = 2 * x  # there is no local scope x
    return y


print(f())
print(x)
l = dict([['a', 1], ['b', 2]])
l
sorted(l, key=lambda x: l[x], reverse=True)  #sorting a dictionary
m = [[3, 1], [5, 2]]
sorted(
    m, key=lambda x: x[1], reverse=True
)  #sorting a nested list based on second element of the nested list
sorted('dsf')
# flexible arguments *args
def f(*args):
    print(type(args))
    print(args[0])
    for i in args:
        print(i)


f(1)
print("")
f(1, 2, 3, 4)
f([1, 2, 3])
def f(**kwargs):
    print(type(kwargs))
    """ print key and value of dictionary"""
    for key, value in kwargs.items():
        print(key, " ", value)


f(country='spain', capital='madrid', population=123456)
def show_details(a, b, *args, **kwargs):
    print("a is ", a)
    print("b is ", b)
    print("args is ", args)
    print("kwargs is ", kwargs)


show_details(1, 2, 3, 4, 5, 6, 7, 8, 9)
print("-----------")
show_details(1, 2, 3, 4, 5, 6, c=7, d=8, e=9)
print("-----------")
def sum(a, b):
    return a + b
sum(1, 2)
num = [1, 2]
sum(*num)
num = {"a": 1, "b": 2}  #keys shouls match with function parameter
sum(**num)
square = lambda x: x**2  # where x is name of argument
print(square(4))
tot = lambda x, y, z: x + y + z  # where x,y,z are names of arguments
print(tot(1, 2, 3))
a = [1, 2, 3]
print(list(map(lambda x: x + 2, a)))
from IPython.display import Image
Image("../input/numpy.png")
l = [1, 2, 3]
a = np.array(l)
a.shape
a
a + 2
l + 2
a2 = np.array([[1, 2], [3, 4]])
a2.shape
a2
a2.shape
np.array([[1, 2], [3, 4]], dtype='float')
a2.astype('float')
np.array(
    [[1, 2, 3], [3, 4]]
)  #due to different numbers in the list, it took the complete list as one element
np.array([[1, 2, 3], [3, 4]]).shape
arr1d_obj = np.array([1, 'a'], dtype='object')
arr1d_obj.shape
arr1d_obj
arr1d_obj.size
a2.tolist()
arr1d_obj.tolist()
list2 = [[1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8]]
arr2 = np.array(list2, dtype='float')
arr2
arr2.shape
arr2.size  #gives number of elements
arr2.ndim  #gives number of dimension
arr2.dtype  #dtype of array
arr2
arr2[:2, :1]  #indexing starts at 0 and follows row, column format
arr2[1]  #second row of the array
arr2[:, 0]  #first column of the array
mask = arr2 > 4  #creating a mask
arr2[mask]  #boolean indexing can be passed but it'll flatten the output
list2 = [[1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8]]
arr2 = np.array(list2, dtype='float')
arr2
arr2[::-1]  #row reversal
arr2[::-1, ::-1]  #row and column reversal
arr2[1, 1] = np.NaN
arr2[1, 2] = np.inf
arr2
np.isnan(arr2)
np.isinf(arr2)
np.isnan(arr2) | np.isinf(arr2)
arr2[np.isnan(arr2) | np.isinf(arr2)] = -1
arr2
arr2.mean(axis=0)  #across rows
arr2.max(axis=1)  #across columns
np.cumsum(
    arr2
)  #this traverse each row completely then shifting to the next row and flattening the output
np.cumsum(arr2, axis=0)  # column wise traversal but retaining the shape
np.cumsum(arr2, axis=1)  # row wise traversal but retaining the shape
arr2.shape
arr2
arr2.reshape(4, 3)
arr2.T  #transpose
arr = np.array([1, 2, 3])
arr
arr.shape
arr[None].shape
a3 = arr2.flatten()
a3[1] = 100
a3
arr2
a3 = arr2.ravel()
a3[1] = 100
a3
arr2
arr2.reshape(6, -1).shape
arr2.resize((4, 3))  #inplace reshaping of array
arr2
arr2
arr1 = arr2.T
arr1
np.dot(arr1, arr2)  #matrix multiplication
np.multiply(arr2, arr2)  #element wise multiplication
np.exp(arr1)  #vectorized exponentiation
a = np.array([1, 2, 3])
a.shape
a + 1
arr2
arr2.shape
a.shape
arr2 + a
np.broadcast_to(a, arr2.shape)
np.arange(4)
np.arange(3, 8)
np.arange(3, 30, 2)  #step size
np.arange(30, 3, -2)  #reverse order
np.linspace(0, 100, num=4)
np.linspace(0, 100, num=4, dtype='int')
np.logspace(0, 3, num=3)
np.zeros([1, 3])
np.ones([1, 3])
a = [1, 2, 3]
np.tile(a, 2)
np.repeat(a, 3)
np.random.rand(
    2, 2
)  #creating random array of given shape and values between 0,1 uniform distibution
np.random.randn(
    2, 2
)  #numbers picked from normal distribution of mean 0 and variance 1 of given shape
np.random.randint(0, 10, [2, 2])  #uniform distribution is used
np.random.random([2, 2])
np.random.choice(
    ['a', 'e', 'i', 'o', 'u'],
    size=10)  #random sample of given size from the list
np.random.choice(
    ['a', 'e', 'i', 'o', 'u'], size=10, p=[0.3, .1, 0.1, 0.4, 0.1]
)  #random sample of given size from the list using predefined probabilities
rn = np.random.RandomState(100)  #seed for reproducability
rn.rand(2, 2), rn.rand(3, 3)
np.random.seed(100)

a = np.random.choice(['a', 'e', 'i', 'o', 'u'], size=10)
a
np.unique(a, return_counts=True)  #getting unique items
arr_rand = np.array([8, 8, 3, 7, 7, 0, 4, 2, 5, 2])
pos = np.where(arr_rand > 5)
arr_rand[pos].shape
arr_rand[pos]
np.take(arr_rand, pos).shape
np.where(
    arr_rand > 5, "a", "b"
)  #like if else: if condition met then first element"a" otherwise 2nd element:"b"
list2 = [[1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8]]
arr2 = np.array(list2, dtype='float')
arr2
np.argmax(arr2, axis=1)
np.argmax(arr2, axis=0)
arr3 = np.ones([3, 4])
arr3
arr2
np.concatenate((arr2, arr3), axis=1)  #concatenate along second axis
arr4 = np.ones([3, 4, 5])
arr5 = np.zeros([3, 4, 5])
arr4
Image("../input/array.png")
np.concatenate(
    (arr4, arr5),
    axis=2)  #concatenate along 3rd axis. remember axis starts at 0
np.concatenate((arr4, arr5), axis=1)
np.concatenate((arr4, arr5), axis=0)
x = np.array([1, 10, 5, 2, 8, 9])
x
sort_index = np.argsort(x)
print(sort_index)
def foo(x):
    if x % 2 == 1:
        return x**2
    else:
        return x / 2
a = np.array([1, 2, 3])
foo_vect = np.vectorize(foo)
foo_vect(a)
x = np.arange(5)
x
a.shape
x[:, np.newaxis]
x[np.newaxis, :]
x[np.newaxis, :, np.newaxis].shape
x
np.clip(
    x, 3, 8
)  #clipping of values in array to a range between minimum and maximum: 3 and 8
Image("../input/regex-example.png")
import re
regex = re.compile('\s+')
text = "Hello World.   Regex is awesome"
regex.split(text)
re.split('\s', text)
text = "101 howard street, 246 mcallister street"
regex_num = re.compile('\d+')  #one or more digits
regex_num.findall(text)
regex_num.split(text)
text2 = "205 MAT   Mathematics 189"
m = regex_num.match(text2)
m.group()
m.start()  #returns the index of the starting
s = regex_num.search(text2)
s.group()
text = """101   COM \t  Computers
205   MAT \t  Mathematics
189   ENG  \t  English"""
regex = re.compile('\s+')
regex.sub(' ', text)  #it replaces the regular expression by ' '
# get rid of all extra spaces except newline
regex = re.compile('((?!\n)\s+)')
print(regex.sub(' ', text))
# define the course text pattern groups and extract
course_pattern = '([0-9]+)\s*([A-Z]{3})\s*([A-Za-z]{4,})'
re.findall(course_pattern, text)
text = "< body>Regex Greedy Matching Example < /body>"
re.findall('<.*>', text)
re.findall('<.*?>', text)
s = re.search('<.*?>', text)  #getting only the first one
s.group()
Image("../input/regex.png")
text = '01, Jan 2015'
print(re.findall('\d{3}', text))
re.findall(r'\btoy\b', 'play toy broke toys')
re.findall(r'\btoy', 'play toy broke toys')
re.findall(r'toy\b', 'play toy broke toys')
re.findall(r'\Btoy\b', 'playtoy broke toys')
re.findall(r'\Btoy\B', 'playtoybroke toys')
re.findall(r'\btoy', 'playtoybroke toys')
emails = """zuck26@facebook.com
page33@google.com
jeff42@amazon.com"""

desired_output = [('zuck26', 'facebook', 'com'), ('page33', 'google', 'com'),
                  ('jeff42', 'amazon', 'com')]
regex = re.compile('([\w]+)@([\w]+).([\w]+)')
regex.findall(emails)
text = """Betty bought a bit of butter, 
But the butter was so bitter, So she bought
some better butter, To make the bitter butter better."""
regex = re.compile('([$bB]\w+)')
regex.findall(text)
sentence = """A, very   very; irregular_sentence"""
desired_output = "A very very irregular sentence"
regex = re.compile('[,\s;_]+')
' '.join(regex.split(sentence))
tweet = '''Good advice! RT @TheNextWeb: What I would do differently if I was learning to code today http://t.co/lbwej0pxOd cc: @garybernhardt #rstats'''
desired_output = 'Good advice What I would do differently if I was learning to code today'
def clean_tweet(tweet):
    tweet = re.sub('http\S+\s*', '', tweet)  # remove URLs
    tweet = re.sub('RT|cc', '', tweet)  # remove RT and cc
    tweet = re.sub('#\S+', '', tweet)  # remove hashtags
    tweet = re.sub('@\S+', '', tweet)  # remove mentions
    tweet = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""),
                   '', tweet)  # remove punctuations
    tweet = re.sub('\s+', ' ', tweet)  # remove extra whitespace
    return tweet


print(clean_tweet(tweet))
