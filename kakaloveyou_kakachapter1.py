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
#dimensional array with 36 element

arr = np.arange(36).reshape(3,4,3)

arr
arr.shape
# With vectorization these operations can be seen as 

# matrix operations which are often more efficient 

# than standard loops

np.random.seed(444)

x = np.random.choice([False, True], size = 100000)

x

def count_transitions(x)-> int:

    count = 0

    for i,j in zip(x[:-1],x[1:]):

        if j and not i:

            count+=1

    return count

count_transitions(x)
np.count_nonzero(x[:-1] < x[1:])
from timeit import timeit

setup = 'from __main__ import count_transitions, x; import numpy as np'

num = 1000

t1 = timeit('count_transitions(x)',setup=setup,number=num)

t2 = timeit('np.count_nonzero(x[:-1] < x[1:])',setup=setup,number=num)

print('Speed difference:{:0.1f}x'.format(t1/t2))
# The operations between two NumPy arrays

a = np.array([1.5,2.5,3.5])

b = np.array([10,5,1])

a/b

