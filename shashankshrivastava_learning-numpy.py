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
import numpy as np
a = np.array([1, 2, 3])
a[2]
help(np.array)
type(a)
b = np.array((3, 4, 5))
type(b)
np.ones( (3,4), dtype=np.int16 )  
np.full( (3,4), 0.11 )
np.arange( 10, 35, 5 )
# it accepts float arguments
np.arange( 0, 2, 0.3 )
np.linspace(0, 5/3, 6)
np.random.rand(2,3)
np.empty((2,3))
A = np.empty((2,3))

a = np.array([1, 2, 5, 7, 8])
a[1:3] = -1
a
#Regular Python array will give error
b = [1, 2, 5, 7, 8]
b[1:3] = -1
b
a = np.array([1, 2, 5, 7, 8])
a_slice = a[1:5]
a_slice[1] = 1000
a

#Original array was modified
a = [1, 2, 5, 7, 8]
a_slice = a[1:5]
a_slice[1] = 1000
a
a_slice
another_slice = a[2:6].copy()
another_slice
#If we modify another_slice, a remains same
a = np.arange(12).reshape(3, 4)
a
#array([[ 0, 1, 2, 3], [ 4, 5, 6, 7], [ 8, 9, 10, 11]])
rows_on = np.array([True, True, True])
rows_on
a[rows_on , : ]      # Rows 0 and 2, all columns
#array([[ 0,  1,  2,  3],
#      [ 8,  9, 10, 11]])
coeffs  = np.array([[2, 6], [5, 3]])
depvars = np.array([6, -9])
solution = np.linalg.solve(coeffs, depvars)
solution
