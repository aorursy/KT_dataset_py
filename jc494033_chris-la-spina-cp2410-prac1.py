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
def is_multiple(n, m):

    if n % m == 0:

        return True

    else:

        return False

is_multiple(10,1)

[2 ** k for k in range(0,9)]
def all_distinct(sequence):

    for i in range(0, len(sequence) - 1):

        for k in range(i + 1, len(sequence)):

            if sequence[i] == sequence[k]:

                return False

    return True



        

all_distinct("1234567")
#When using yield I got an error - I tried to look at solutions and still couldn't fix it. 



def harmonic_generator(n):

    h = 0

    for i in range(1, n + 1):

        h += 1 / i

#         yield h

        print(h)



harmonic_generator(3)
