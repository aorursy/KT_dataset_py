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
def is_multiple(n,m):

    return n % m == 0
[2 ** i for i in range(0,9)]

def all_distinct(seq):

    for i in range (0, len(seq)-1):

        for j in range (i+1,len(seq)):

            if seq [i] == seq[j]:

                return False

    return True
def harmonic_gen(n):

    h = 0

    for i in range(1, n+1):

        h += 1/i 

        yield h