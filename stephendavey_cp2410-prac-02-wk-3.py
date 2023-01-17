# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import math

n = 100000

ln = math.log(n)

print(2**10) # Big O(1)

print(ln) # Big O logn

print(2*ln) # Big O logn

print()

print(3*n + 100*ln) # Big O(n)

print(4*n) # Big O(n)

print()



print(n*ln) # Big O(nlogn)

print(4*n*ln + 2*n) # Big O(nlogn)

print(n**2 + 10*n) # Big O(n**2)

print(n**3) # Big O(n**3)

# print(2**n) # Big O(2**n)

n = 1.24

logn = math.log(n, 2)

a = round(8*n*logn, 2)

b = round( 2*(n**2),2)

print(f"For n = {n}")

print(f"A = {a} and B = {b}")

# Let number of operations be integers

# a >= b for n >= 2
def example1(s):

    """Return the sum of the elements in sequence s"""

    n = len(s) # 1 primitive operation

    total = 0 # 1

    for j in range(n):

        total += s[j] # 2 ops n times = 2n

    return total # 1

    
def example2(s):

    """Return sum of elements with even index in s"""

    n = len(s) # 1

    total = 0 # 1

    for j in range(0, n, 2): 

        total += s[j] # 2ops, at worst n/2 times = n times

        # Obviously, for odd n the algorithm performs slightly better, (n-1) being the dominant factor

    return total # 1
def example3(s):

    """Return sum of prefix sums of s"""

    n = len(s) # 1

    total = 0 # 1

    for j in range(n):

        for k in range(1 + j): #(1+j)  n times from outer loop

            total += s[k] # 2 ops, 1+2+3+...+n times ; arthimetic series = n(n+1)/2

            # 2 ops, n(n+1)/2 times = n(n+1)

    return total # 1
def example3a(s):

    """Example 3 with a list comprehension"""

    n = len(s)

    total = 0

    for j in range(n):

        total += sum([s[k] for k in range(1+j)])

    return total



# Playing with some tests

from time import time

s = [0 * 10000000]

start = time()

print(example3(s))

stop = time()

print(f'elapsed time : {stop - start}s')

start = time()

print(example3a(s))

stop = time()

print(f'elapsed time : {stop - start}s')

def example4(s):

    """Return sum of prefix sums of s"""

    n = len(s) # 1

    prefix = 0 # 1

    total = 0 # 1

    for j in range(n):

        prefix += s[j]

        total += prefix 

        # 4 ops, n times

    return total # 1
def example5(a, b):

    """Return number of elements in B equal to the sum of prefix sums in A"""

    n = len(a) # 1

    count = 0 # 1

    for i in range(n):

        total = 0 # n times

        for j in range(n):

            for k in range(1 + j): # 1+j called n^2 times

                total += a[k] # 2 ops, n(n+1)/2 * n = n^3 + n^2

        if b[i] == total: # n times

            count += 1  # up to n times

    return count # 1