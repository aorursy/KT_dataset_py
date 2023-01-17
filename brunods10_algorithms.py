# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import matplotlib.pyplot as plt # data visualization
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def check_answer(qnum, answer):
    file = 'https://s3.amazonaws.com/brunodos3.com/test.csv'
    df = pd.read_csv(file, index_col=[0], header=None)
    answer_true = df.loc[qnum].values[0]
    if str(answer) == answer_true:
        is_correct = 'Correct!'
    else:
        is_correct = 'Not correct.'
    return is_correct

file = 'https://s3.amazonaws.com/brunodos3.com/test.csv'
df = pd.read_csv(file, index_col=[0], header=None)
# load a test file to dataframe
test_file = 'https://s3.amazonaws.com/brunodos3.com/mtcars.csv'
df = pd.read_csv(test_file)
# print dataframe
df.shape
is_true = df['mpg'] > 25
df.loc[is_true, :]
check_answer(1, 'Hello world!')
file = 'https://s3.amazonaws.com/brunodos3.com/mtcars.csv'
df = pd.read_csv(file)
df.head()
# answer here




check_answer(2, 'your answer here')
alist = [1, 2, 3, 4, 5, 1234, 43]
for i in alist:
    if i % 2 == 0:
        print(i, 'is even')
    else:
        print(i, 'is not even')
file = 'https://s3.amazonaws.com/brunodos3.com/mtcars.csv'
df = pd.read_csv(file)
df.head()
# your answer here



check_answer(3, 10)
# finding the mean using basic Python
def calc_mean(alist):
    total = 0  # total sum of list
    n = len(alist)  # length of list
    for i in alist:
        total += i  # add list element to total
    mean = total / n  # calculate mean
    return mean

print('The mean is:', calc_mean(df['disp']))
# finding the mean using Numpy
print('The mean is:', np.mean(df['disp']))
# finding the mean using Pandas
print('The mean is:', df['disp'].mean())
file = 'https://s3.amazonaws.com/brunodos3.com/mtcars.csv'
df = pd.read_csv(file)
# your answer here



check_answer(3, 10)
df['mpg'].max() - df['mpg'].min()
# your answer here



check_answer(3, 10)
def fib(n):
    '''
    Returns the nth Fibonacci number.
    
    Base case: when n <= 1, fib(n) == 1
    Recursive case: fib(n) = fib(n - 1) + fib(n - 2)
    
    '''
    if n <= 1:  # base case        
        return n
    else:  # recursive case
        num = fib(n - 1) + fib(n - 2)
        return num

for i in range(1, 11):
    print(f'#{i}\t', fib(i))
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)
factorial(50)
# your answer here



check_answer(6, 10)
alist = [1, 1, 2, 2, 3, 4, 4, 5, 6, 7, 7, 7]
blist = [1, 2, 2, 8, 8, 8, 9, 10]
print(set(alist))
print(set(blist))

# common numbers between alist and blist
print('intersection')
print(set(alist).intersection(blist))

# all unique numbers between alist and blist
print('union')
print(set(alist).union(blist))

# difference
print('differences')
print(set(alist) - set(blist))
print(set(blist) - set(alist))

# numbers that are not shared between sets
print('symmetric difference')
print(set(alist) ^ set(blist))
from itertools import combinations
list(combinations([1, 2, 3, 4], 3))
file = 'https://s3.amazonaws.com/brunodos3.com/add_to_zero.csv'
df_num = pd.read_csv(file, delimiter=',', header=None)
df_num
np.random.binomial(n=1, p=0.5, size=1000).mean()
# your answer here



check_answer(8, 10)
import string
import random
def random_str(n):
    astr = ''
    for i in range(n):
        aletter = random.choice(string.ascii_uppercase)
        astr += aletter
    return astr
        
def str_shape(astr, x):
    astr[i:x]
vec = np.arange(0, 101)
print(vec[::10])
print(vec[::-10])
file = 'https://s3.amazonaws.com/brunodos3.com/word_search.txt'
word_search = pd.read_table(file, header=None)[0][0]
word_search
agraph = {
    'a': ['b', 'c'],
    'b': ['d', 'e'],
    'c': ['f', 'g'],
    'd': ['h'],
    'e': ['i', 'j'],
    'f': ['k'],
    'g': [],
    'h': [],
    'i': [],
    'j': [],
    'k': []
}
# creating list
alist = np.arange(0, 10)
[i for i in alist if i % 2 == 0]  # comprehension that only gets even numbers from a list
# creating dictionary
blist = np.arange(10, 20)
{k:v for k, v in zip(alist, blist)}
file = 'https://s3.amazonaws.com/brunodos3.com/network_nodes.txt'
network = pd.read_csv(file, header=None)
network.head()
dict_net = {i.split()[0]:i.split()[1] for i in network[0]}
dict_net