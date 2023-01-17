# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



print("Question 1")

def is_multiple(n,m):

    if n % m == 0:

        return True

    else:

        return False

print(is_multiple(8,2))

print("")



print("Question 2")

indiciesOfTwo = [2**x for x in range(0, 9)]

print(indiciesOfTwo)

print("")



print("Question 3")

def all_distinct(seq):

    for i in range(0, len(seq) - 1):

        for j in range(i + 1, len(seq)):

            if seq[i] == seq[j]:

                return False

    return True

sequence = [3, 6, 4, 5, 1, 7, 8, 9]

print(all_distinct(sequence))

print("")



print("Question 4")

def harmonic_list(n):

    result = []

    h = 0

    for i in range(1, n + 1):

        h += 1/i

        result.append(h)

    return result

def harmonic_gen(n):

    h = 0

    for i in range(1, n + 1):

        h += 1/i

        yield h

test1 = 5

print(harmonic_list(test1))

test2 = harmonic_gen(5)

print(list(test2))

print("")



# Any results you write to the current directory are saved as output.