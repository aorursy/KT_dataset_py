import numpy as np

import pandas as pd

import sys

import string

from datetime import datetime
# n = 999

# p = 999

# %timeit n**p

# %timeit pow(n, p)

# %timeit np.power(n,p)



# Extreme test

# %time 2**sys.maxsize
# print(sys.maxsize**9999)
df = pd.DataFrame(dtype=object)
a=95800

b=217519

c=414560

d=422481
range_from=1

range_to=10000



rng = np.arange(range_from, range_to)

def multiple(n): return [n * v for v in rng]



df['a'] = multiple(a)

df['b'] = multiple(b)

df['c'] = multiple(c)

df['d'] = multiple(d)
df.drop(df.index, inplace=True)
def letters(a=26): # amount a

    if a < 26:

        letters = list(string.ascii_lowercase[:a])

    elif (a**2)<(26**2):

        letters = list(string.ascii_lowercase[:26])        

        letters.extend([i+b for i in letters [:(a-26)] for b in letters]) # maximum :26*26

    else:

        raise ValueError(f'Input amount={a} too high. Maximum is {26+26*26}.')

    return letters
from itertools import permutations

b = 1

e = 2

r = range(b,e+1)



p = permutations(r) # permutations p is faster than combinations c because it doesn't order



c = combinations(r) # combinations c



for e in r:

    print(e)



for e in p:

    print(e)
a = 20 # amount of variables a

m = 0 # memorize m; up to where the iteration already happened with another variable



b = 101000 # beginning b of range to iterate through; no solution found below the threshold of 95700 for 4 variables

e = 100000 # end e of range to iterate through



    

# inefficient code down below; start b where a ends; start c where b ends to avoid double combinations, e.g. 1+2=2+3

for idx, letter in enumerate(letters(a)):

    b = e - m # avoid double combinations leading to the same sum, every sum should be only handled once

    df[letter] = pd.Series(range(b,e), dtype=object)

    if idx > 2:

        m = e
p = pd.DataFrame(dtype=object) # potencies p

p.drop(p.index, inplace=True)
# idea: also take from memory if factor *10**x = *10^x is included, because then the first potency rule applies, include in-memory optimization for potency rules
e = 4

rng_srt = 2**1*10**1 # range start

rng_end = 2**1*10**1+1000 # range end # idea: if rng_srt or rng_end > sys.maxsize: dynamically split into parts of a maximum of 200 values, write and then take next package of values to make sure you save your result 



p = pd.DataFrame(dtype=object) # potencies p



p['b'] = pd.Series(range(rng_srt,rng_end+1), dtype=object) # base b

p['e'] = e # exponent e

p['v'] = [value**e for value in p['b']] # value v



p.tail()
memory = 'primes_memory' + datetime.now() # create file path to memory of primes



p.to_csv(memory); # write to harddrive and supress output via ;
def load_csv():

    loaded = bool(False) # only load CSV once

    

    if loaded is False:  

        try:

            p = pd.read_csv(memory, sep=',', header=0) #http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html

            loaded = bool(True)

        except:

            !ls

            raise ValueError(f'File "{m}" could not be loaded. Does it exist?')       



def power_from_memory():

    load_csv()



    df['name'] # would select a column called name

    # This would show observations which start with STARBUC

    match = df['name'].str.contains('(^STARBUC)')

    print(df['name'][match].value_counts())



power_from_memory()
import string

def letters(a=26): # amount a

    if a < 26:

        letters = list(string.ascii_lowercase[:a])

    elif (a**2)<(26**2):

        letters = list(string.ascii_lowercase[:26])        

        letters.extend([i+b for i in letters [:(a-26)] for b in letters]) # maximum :26*26

    else:

        raise ValueError(f'Input amount={a} too high. Maximum is {26+26*26}.')

    return letters
def calculate():

    df['a^4'] = [n**4 for n in df['a']]

    df['b^4'] = [n**4 for n in df['b']]

    df['c^4'] = [n**4 for n in df['c']]

    df['d^4'] = [n**4 for n in df['d']]



def check():

    df['result_is_positive'] = df.apply(lambda e: (True if e['d^4'] > 0 else False), axis=1) # result is positive

    %timeit df['is_solution'] = df.apply(lambda e: (True if(e['a^4']+e['b^4']+e['c^4']==e['d^4']) else False), axis=1) # formula is true for input parameters

    %timeit df['is_solution'] = df.apply(lambda e: (True if(e['a^4']+e['b^4']==e['d^4']-e['c^4'])) # equivalent transformation: minus c^4 on both sides; should be more efficient

    df['equation_is_true'] = df.apply(lambda e: (True if(e['result_is_positive'] and e['is_solution']) else False), axis=1) # equation is true if both above conditions are true

    df['entry_disproves_proposition'] = df.apply(lambda e: -e['equation_is_true'], axis=1)



calculate()

check()
df.head()
df.to_csv()
df.loc[df['entry_disproves_proposition'] == True]
!ls