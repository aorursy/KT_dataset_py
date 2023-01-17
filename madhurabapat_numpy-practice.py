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

a = np.arange(1,101)

print(a)
a = np.arange(2,101,2)

print(a)



b = np.arange(1,101,2)

print(b)
a = np.arange(1,51)

print(a)

b = a.reshape(10,5)

print("Reshaped array:")

print(b)
import numpy as np

a = np.arange(1,37)

print(a)

arr = a.reshape(3,3,4)

print(arr)
print("Number of dimensions/axis of the ndarray:")

print(arr.ndim)



print("Minimum value in the whole array:",(arr.min()))                   



print("Minimum value along the axis 0:")

print(arr.min(axis=0))



print("Minimum value along the axis 1:")

print(arr.min(axis=1))



print("Minimum value along the axis 2:")

print(arr.min(axis=2))
print("Maximun value in the whole array:",(arr.max()))                   



print("Maximun value along the axis 0:")

print(arr.max(axis=0))



print("Maximun value along the axis 1:")

print(arr.max(axis=1))



print("Maximun value along the axis 2:")

print(arr.max(axis=2))
mat = np.array(np.arange(1,10))

mat = mat.reshape(3,3)

#mat = np.array([[1,2],[3,4]])

inve = np.linalg.inv(mat) 

print("Matrix :")

print(mat) 

print("Inverse Matrix :")

print(inve)

print("Dot product Matrix :")

print((mat.dot(inve)))
mat = np.array([[5,11,15], [4, -12, 3], [-2,8,17]])

print(mat)

print("Determinent:",np.linalg.det(mat))
A = np.array([[1, 2,-1], [2, 1, 1], [1, 2, 1]])

print("Matrix A :")

print(A)



B = np.array([4, -2, 2])

print("Matrix B :")

print(B)





inv_A = np.linalg.inv(A)

print("Inverse of Matrix A :")

print(inv_A)





X = np.linalg.inv(A).dot(B)

print("Values of x and y and Z of the equations :")

print(X)