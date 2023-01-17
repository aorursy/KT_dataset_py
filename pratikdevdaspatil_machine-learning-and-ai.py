# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
a= np.array([1,2,3,4,5])
print(a)

import numpy as np
a= ([1,2,3],[4,5,6],[7,8,9])
j=np.array(a)
print(j)
import numpy as np

bool_arr = np.array([1, 0.5, 0, None, 'a', '', True, False], dtype=bool)
print(bool_arr)
import numpy as np
a = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
print (a)
rows = a.shape[0]
cols = a.shape[1]
print (rows)
print (cols)
for x in range(0, cols -1):
    for y in range(0, rows -1):
        z=a[x,y]
        if z%2!=0:
            print(z,"odd")
            
            
        
       
    




import numpy as np
x = np.array([[ 0.42436315, 0.48558583, 0.32924763], [ 0.7439979,0.58220701,0.38213418], [ 0.5097581,0.34528799,0.1563123 ]])
print("Original array:")
print(x)
print("Replace all elements of the said array with .5 which are greater than .5")
x[x > .5] = .5
print(x)

import numpy as np
array1 = np.array([0, 10, 20, 40, 60])
print("Array1: ",array1)
array2 = [10, 30, 40]
print("Array2: ",array2)
print("Common values between two arrays:")
print(np.intersect1d(array1, array2))
import numpy as np
array1=np.array(1,2,3,4,5)
array2=np.array(4,5,9,10,11)
z=np.diff(array1,array2)
print(z)
import numpy

a = numpy.array([0, 1, 2, 3, 4, 5, 6])
b = numpy.array([6, 5, 4, 3, 2, 1, 6])
numpy.where(a==b)