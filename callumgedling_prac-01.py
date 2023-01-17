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
n = 4

m = 2

def is_multiple(m,n):

    if n%m == 0:

           return True 

    else:

        return False

is_multiple(m,n)

[2 ** i for i in range (0,9)]
sequence = [1, 2, 3, 4, 5, 4]

def check_all_different(sequence):

    for i in range(0, len(sequence) -1):

        for j in range(i + 1, len(sequence)):

            if sequence[i] == sequence[j]:

                return False

    return True



check_all_different(sequence)

        
n = 3

def harmonic_gen(n):

    h = 0

    for i in range(1, n+1):

        h += 1/i

        yield h

harmonic_gen(n)