# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def example1(S):

    """Return the sum of the elements in sequence S."""

    n = len(S)

    total = 0

    for j in range(n): # loop from 0 to n-1

        total += S[j]

    return total
def example2(S):

    """Return the sum of the elements with even index in sequence S."""

    n = len(S)

    total = 0

    for j in range( 0 , n, 2 ): # note the increment of 2

        total += S[j]

    return total
def example3(S):

    """Return the sum of the prefix sums of sequence S."""

    n = len(S)

    total = 0

    for j in range(n): # loop from 0 to n-1

        for k in range( 1 + j): # loop from 0 to j

            total += S[k]

    return total
def example4(S):

    """Return the sum of the prefix sums of sequence S."""

    n = len(S)

    prefix = 0

    total = 0

    for j in range(n):

        prefix += S[j]

        total += prefix

    return total
def example5(A, B): # assume that A and B have equal length

    """Return the number of elements in B equal to the sum of prefix sums in A."""

    n = len(A)

    count = 0

    for i in range(n): # loop from 0 to n-1

        total = 0

        for j in range(n): # loop from 0 to n-1

            for k in range( 1 + j): # loop from 0 to j

                total += A[k]

            if B[i] == total:

                count += 1

    return count