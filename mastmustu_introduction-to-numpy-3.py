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
# Singular-value decomposition

from numpy import array

from scipy.linalg import svd

# define a matrix

A = array([[1, 2], [3, 4], [5, 6]])

print(A)

# remember m = 3 and n = 2 

# SVD

U, s, VT = svd(A)

print(U)  # U should 3 x 3

print(s)  

print(VT) # V should be 2 x 2
%%time

# Singular-value decomposition

from numpy import array

from scipy.linalg import svd

# define a matrix

A = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(A)

# remember m = 3 and n = 3 

# SVD

U, s, VT = svd(A)

print(U)  # U should 3 x 3

print(s)  

print(VT) # V should be 3 x 3


import numpy as np 



x = np.array([[1,2],[3,4]]) 

y = np.linalg.inv(x) 

print("Original Matrix :")

print(x) 

print("Inverse Matrix :")

print(y)

print("Dot product Matrix :")

print((np.around(np.dot(x,y),2)))
import numpy as np

a = np.array([[1,2], [3,4]]) 

print(a)

print(np.linalg.det(a))



#(1*4 - 2*3) == -2

%%time 

import numpy as np 



b = np.array([[6,1,1], [4, -2, 5], [2,8,7]]) 

print(b) 

print(np.linalg.det(b) )

print(6*(-2*7 - 5*8) - 1*(4*7 - 5*2) + 1*(4*8 - -2*2)) # mathemical calculation
import numpy as np



m_list = [[4, 3], [-5, 9]]

A = np.array(m_list)

print("Matrix A :")

print(A)



B = np.array([20, 26])

print("Matrix B :")

print(B)



inv_A = np.linalg.inv(A)

print("Inverse of Matrix A :")

print(inv_A)





X = np.linalg.inv(A).dot(B)

print("Values of x and y of the equations :")

print(X)
import numpy as np



A = np.array([[3, 1], [2, 2]])

print("Matrix A:")

print(A)



w, v = np.linalg.eig(a)



print( "Eigen Value :")

print(w)

print("Eigen Vector :") 

print(v)
%%time

import numpy as np



#A = np.array([[1,2,3], [4, 5,6],[7,8,9]])

A = np.arange(1,10001)

A = A.reshape(100,100)

print("Matrix A:")

print(A)



w, v = np.linalg.eig(A)



print( "Eigen Value :")

print(w)

print("Eigen Vector :") 

print(v)