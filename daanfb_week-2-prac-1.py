# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))





# CP2410 Prac 1 Week 2 Questions:



#1. (R-1.1) Write a short Python function, is_multiple( n, m), that takes two integer values and returns

#True if n is a multiple of m, that is, n = mi for some integer i, and False otherwise.



def is_multiple(n, m):

    if m % n == 0:

        print("N is a multiple of M")

    else:

        print("N is not a multiple of M")



# Test for True

is_multiple(5, 10)



# Test for False

is_multiple(6, 10)





# 2. (R-1.11) Demonstrate how to use Python’s list comprehension syntax to produce the list [1, 2, 4, 8,16, 32, 64, 128, 256].



n = 9

squares = [2 ** k for k in range(0, n)]

print(squares)





# 3. (C-1.15) Write a Python function that takes a sequence of numbers and determines if all the

# numbers are different from each other (that is, they are distinct).



def test_distinct(seq):

    if len(seq) == len(set(seq)):

        print("Numbers are distinct")

    else:

        print("Numbers are not distinct")





test_distinct([1, 2, 3, 4])

test_distinct([1, 1, 2, 3, 4])





# 4. The n-th harmonic number is the sum of the reciprocals of the first n natural numbers. For example,

# H3

#  = 1 + ½ + ⅓ = 1.833. A simple Python function for creating a list of the first n harmonic numbers

# follows:

# def harmonic_list(n):

# result = []

# h = 0

# for i in range(1, n + 1):

# h += 1 / i

# result.append(h)

# return result

# Convert this function into a generator harmonic_gen(n) that yields each harmonic number.



def harmonic_gen(n):

    h = 0

    for i in range(1, n + 1):

        h += 1 / i

        yield h




















