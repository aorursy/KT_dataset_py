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
# Initialize a Numpy array of zeros that has the length of total unique feautures/properties (10 in this case)
a = np.array([0,0,0,0,0,0,0,0,0,0])
# Example numpy array of a compound, a row in the pandas dataframe, listing all properties for this compound
b = np.array([2,4,6,8])
# This numpy array is an example of the unique properties that we actually care about
c = np.array([1,2,4,6,7,9])
# So what we are trying to do here is output an array of length 10 for every row in the pandas dataframe, with either
# a "0" indicating that this compound (row) does not have this property (column), or a "1" indicating that it does.
# Here the a[0] element represents unique property #1 that this row/compound has or does not have
# So properties 1-10 are represented by this initialized array (not 0-9 for example).
a
b
c
for x in np.nditer(b):
    for y in np.nditer(c):
        if x == y:
            a[x-1] = 1

print(a)
# Test is successful!  What we have demonstrated here is that the zeroed array "a" now has the binary representation 
# of the unique feature set of properties that we care about (from array "c") as found in array "b".
