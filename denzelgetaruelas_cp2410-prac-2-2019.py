# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



"""

Question 1

2^10 = O(1)

2logn = O(log n)

3n + 100 log n = O(n)

4n = O(n)

n log n = O(n log n)

4n log n + 2n = O(n log n)

n^2 + 10n = O(n^2)

n^3 = O(n^3)

2^n = O(2^n)

"""



"""

Question 2

Simplifying A and B will give us:

    4 log n = n

Where A is (4 log n) and B is (n)

By plugging in values for n (i.e. exponents of 2),

A will always be greater than B up until n = 16.

Therefore, A will run faster when n is greater or equal to 16.

"""



"""

Question 3

Using the definition of the big Oh notation,

since d(n) = O(f(n)), we have positive c and n0

a*d (n) <= a*c*f(n) for n => n0

give c2 = a*c

a*d(n) <= c2f(n) for all n => n0

That makes a*d(n) = O(f(n))

"""



"""

Question 4

"""

def example1(S):

    # Return the sum of the elements in sequence S.

    n = len(S)

    total = 0

    for j in range(n): # loop from 0 to n-1

        total += S[j]

    return total



print("example1")

list1 = [2, 3, 4, 5, 6]

print("list: " + str(list1))

print("Sum of the elements: " + str(example1(list1)))

print()



"""

2 operation before the loop,

'n = len(S)' and 'total = 0'

In the for loop, 3 operations for every time the loop runs

Therefore, the overall run time is 2 + 3n = O(n)

"""



def example2(S):

    # Return the sum of the elements with even index in sequence S.

    n = len(S)

    total = 0

    for j in range(0, n, 2): # note the increment of 2

        total += S[j]

    return total



print("example2")

list2 = [2, 6, 4, 5, 6, 8]

print("list: " + str(list2))

print("Sum of the elements with even index: " + str(example2(list2)))

print()



"""

2 operations before the loop,

'n = len(S)' and 'total = 0'

Loop runs n/2 times, which gives 1.5n

"""



def example3(S):

    # Return the sum of the prefix sums of sequence S.

    n = len(S)

    total = 0

    for j in range(n): # loop from 0 to n-1

        for k in range(1 + j): # loop from 0 to j

            total += S[k]

    return total



print("example3")

list3 = [2, 3, 4, 5, 6, 8]

print("list: " + str(list3))

print("Sum of the prefix sums: " + str(example3(list3)))

print()



"""

2 operations before the loop,

'n = len(S)' and 'total = 0'

Inner loop runs 1 + 2 + ... + n times, with 3 primitive operations

Overall runtime is 2 + 1.5n(n + 1) = 2 + 1.5n^2 + 1.5n = O(n^2)

"""



def example4(S):

    # Return the sum of the prefix sums of sequence S

    n = len(S)

    prefix = 0

    total = 0

    for j in range(n):

        prefix += S[j]

        total += prefix

    return total



print("example4")

list4 = [2, 3, 4, 5, 6, 8, 9, 10]

print("list: " + str(list4))

print("Sum of the prefix sums: " + str(example4(list4)))

print()



"""

3 operation before the loop,

'n = len(S)', 'prefix = 0' and 'total = 0'

Loop runs 5 primitive operations each time.

Therefore, overall runtime is 3 + 5n = O(n)

"""



def example5(A, B): # assume that A and B have equal length

    # Return the number of elements in B equal to the sum of prefix sums in A.

    n = len(A)

    count = 0

    for i in range(n): # loop from 0 to n-1

        total = 0

        for j in range(n): # loop from 0 to n-1

            for k in range(1 + j): # loop from 0 to j

                total += A[k]

        if B[i] == total:

            count += 1

    return count



print("example5")

list5a = [2, 6]

list5b = [10, 2]

print("list1: " + str(list5a))

print("list2: " + str(list5b))

print("Number of elements in list2 to the sum of the prefix sums in list1: " + str(example5(list5a, list5b)))



"""

Running time is 2 + n(1 + 1.5n(n + 1) + 5) = O(n^3)

"""



# Any results you write to the current directory are saved as output.