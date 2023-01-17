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
cities = pd.read_csv("../input/cities.csv")
print(cities.CityId.unique().shape[0])
# TL; DR

def sieve_of_eratosthenes(N):
    # Complexity: O(NloglogN)
    sieve = [True] * (N+1)
    sieve[:2] = [False] * 2
    for i in range(2, N+1):
        if sieve[i]:
            for j in range(2*i, N+1, i):
                sieve[j] = False
    P = [i for i in range(2, N+1) if sieve[i]]
    P_set = set(P)
    
    def is_prime_inset(x):
        # Complexity: O(1)
        nonlocal P_set
        return x in P_set
    return is_prime_inset

is_prime = sieve_of_eratosthenes(200000)
cities["CityId"].apply(is_prime).head(30)
# define functions
from math import sqrt
from bisect import bisect

def is_prime_sqrt(x):
    # Complexity: O(sqrt(N))
    if x < 2:
        return False
    elif x < 4:
        return True
    for a in range(2, int(sqrt(x))):
        if x % a == 0:
            return False
    else:
        return True

def sieve_of_eratosthenes(N):
    # Complexity: O(NloglogN)
    sieve = [True] * (N+1)
    sieve[:2] = [False] * 2
    for i in range(2, N+1):
        if sieve[i]:
            for j in range(2*i, N+1, i):
                sieve[j] = False
    P = [i for i in range(2, N+1) if sieve[i]]
    P_set = set(P)

    def is_prime_sieve(x):
        # Complexity: O(1)
        nonlocal sieve
        return sieve[x]

    def is_prime_inlist(x):
        # Complexity: O(N)
        nonlocal P
        return x in P

    def is_prime_inset(x):
        # Complexity: O(1)
        nonlocal P_set
        return x in P_set
    
    def is_prime_prime_sqrt(x):
        """This function can check up to N ^ 2"""
        nonlocal P
        if x < 2:
            return False
        elif x < 4:
            return True
        for p in P[:bisect(P, sqrt(x)+1)]:
            if x % p:
                return False
        else:
            return True
    return is_prime_sieve, is_prime_inlist, is_prime_inset, is_prime_prime_sqrt, P

is_prime_sieve, is_prime_inlist, is_prime_inset, is_prime_prime_sqrt, P = sieve_of_eratosthenes(200000)
np.random.seed(2434)
randints = np.random.randint(0, 197769, 200000)

def calc_func(func, randints):
    for x in randints:
        func(x)
# https://www.kaggle.com/c/traveling-santa-2018-prime-paths/discussion/72176
from sympy.ntheory.primetest import isprime as isprime_sympy
print("sympy")
%timeit calc_func(isprime_sympy, randints)

# my functions
print("is_prime_prime_sqrt")
%timeit calc_func(is_prime_sqrt, randints)
print("is_prime_prime_sieve")
%timeit calc_func(is_prime_sieve, randints)
print("is_prime_prime_inlist")
%timeit calc_func(is_prime_inlist, randints[:2000])
print("is_prime_prime_inset")
%timeit calc_func(is_prime_inset, randints)

print("is_prime_prime_sqrt")
%timeit calc_func(is_prime_prime_sqrt, randints)

